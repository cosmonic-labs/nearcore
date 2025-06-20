use crate::errors::ContractPrecompilatonResult;
use crate::logic::errors::{
    CacheError, CompilationError, FunctionCallError, MethodResolveError, VMLogicError,
    VMRunnerError, WasmTrap,
};
use crate::logic::{
    Config, ExecutionResultState, External, GasCounter, MemSlice, MemoryLike, VMContext, VMLogic,
    VMOutcome,
};
use crate::runner::VMResult;
use crate::{
    CompiledContract, CompiledContractInfo, Contract, ContractCode, ContractRuntimeCache,
    NoContractRuntimeCache, get_contract_cache_key, imports, lazy_drop, prepare,
};
use near_parameters::RuntimeFeesConfig;
use near_parameters::vm::{LimitConfig, VMKind};
use std::borrow::Cow;
use std::cell::{RefCell, UnsafeCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, LazyLock, Mutex, RwLock};
use wasmtime::{
    DEFAULT_INSTANCE_LIMIT, DEFAULT_MEMORY_LIMIT, DEFAULT_TABLE_LIMIT, Engine, Extern, Instance,
    Linker, Memory, MemoryType, Module, Store, Strategy,
};
use wasmtime::{ExternType::Func, ModuleExport};

const GUEST_PAGE_SIZE: usize = 1 << 16;

static HOST_PAGE_SIZE: LazyLock<usize> = LazyLock::new(|| {
    #[cfg(miri)]
    {
        4096
    }
    #[cfg(unix)]
    {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE).try_into().unwrap() }
    }
    #[cfg(windows)]
    unsafe {
        let mut info = core::mem::MaybeUninit::uninit();
        winapi::um::sysinfoapi::GetSystemInfo(info.as_mut_ptr());
        info.assume_init_ref().dwPageSize as _;
    }
});

//const MAX_CONCURRENCY: u16 = 1 << 11;

type Caller = wasmtime::Caller<'static, ()>;
thread_local! {
    pub(crate) static CALLER: RefCell<Option<Caller>> = const { RefCell::new(None) };
}

pub struct WasmtimeMemory(Memory);

struct InstanceContext {
    store: Store<()>,
    memory: WasmtimeMemory,
    instance_budget: usize,
    table_budget: usize,
    memory_budget: usize,
}

//struct InstancePermit<'a> {
//    resources: ResourcesRequired,
//    instances: &'a AtomicU64,
//    memories: &'a AtomicU64,
//    tables: &'a AtomicU64,
//}
//
//impl Drop for InstancePermit<'_> {
//    fn drop(&mut self) {
//        self.instances.fetch_sub(1, Ordering::Release);
//        self.memories.fetch_sub(self.resources.num_memories.into(), Ordering::Release);
//        self.tables.fetch_sub(self.resources.num_tables.into(), Ordering::Release);
//    }
//}
//
//// A simple semaphore implemented using a spinlock.
//// It is not expected to be contended often
//#[derive(Clone, Default)]
//struct ConcurrencySemaphore {
//    instances: Arc<AtomicU64>,
//    memories: Arc<AtomicU64>,
//    tables: Arc<AtomicU64>,
//}
//
//impl ConcurrencySemaphore {
//    fn acquire(&self, resources: ResourcesRequired) -> InstancePermit<'_> {
//        debug_assert!(resources.num_memories <= MAX_CONCURRENCY.into());
//        debug_assert!(resources.num_tables <= MAX_CONCURRENCY.into());
//        while self.instances.fetch_add(1, Ordering::Acquire) > MAX_CONCURRENCY.into() {
//            self.instances.fetch_sub(1, Ordering::Release);
//            hint::spin_loop();
//        }
//        while self.memories.fetch_add(resources.num_memories.into(), Ordering::Acquire)
//            > MAX_CONCURRENCY.into()
//        {
//            self.memories.fetch_sub(resources.num_memories.into(), Ordering::Release);
//            hint::spin_loop();
//        }
//        while self.tables.fetch_add(resources.num_tables.into(), Ordering::Acquire)
//            > MAX_CONCURRENCY.into()
//        {
//            self.tables.fetch_sub(resources.num_tables.into(), Ordering::Release);
//            hint::spin_loop();
//        }
//        return InstancePermit {
//            resources,
//            instances: &self.instances,
//            memories: &self.memories,
//            tables: &self.tables,
//        };
//    }
//}

impl InstanceContext {
    fn new(engine: &Engine, config: &LimitConfig) -> Self {
        let mut store = Store::new(engine, ());
        let mut minimum = config
            .initial_memory_pages
            .try_into()
            .unwrap_or(usize::MAX)
            .saturating_mul(GUEST_PAGE_SIZE);
        let maximum = config
            .max_memory_pages
            .try_into()
            .unwrap_or(usize::MAX)
            .saturating_mul(GUEST_PAGE_SIZE);
        let mut initial_memory_pages = config.initial_memory_pages;
        while minimum < maximum {
            let host_rem = minimum % *HOST_PAGE_SIZE;
            let guest_rem = minimum % GUEST_PAGE_SIZE;
            if host_rem == 0 && guest_rem == 0 {
                // Ensure that the mmap-ed memory region size is a multiple of host page size to ensure CoW is used
                // https://github.com/bytecodealliance/wasmtime/blob/18b42ef4e48e498026237013df8ed5af9da0d72d/crates/wasmtime/src/runtime/vm/memory.rs#L526-L530
                initial_memory_pages =
                    u32::try_from(minimum.saturating_div(GUEST_PAGE_SIZE)).unwrap_or(u32::MAX);
                break;
            }
            minimum = minimum.saturating_add(host_rem.max(guest_rem)).min(maximum);
        }
        let memory = Memory::new(
            &mut store,
            MemoryType::new(initial_memory_pages, Some(config.max_memory_pages)),
        )
        .expect("failed to construct memory");
        Self {
            store,
            memory: WasmtimeMemory(memory),
            instance_budget: DEFAULT_INSTANCE_LIMIT,
            table_budget: DEFAULT_TABLE_LIMIT,
            memory_budget: DEFAULT_MEMORY_LIMIT,
        }
    }
}

static VMS: LazyLock<RwLock<HashMap<Arc<Config>, WasmtimeVM>>> = LazyLock::new(RwLock::default);
static INSTANCE_CONTEXTS: LazyLock<Mutex<HashMap<Arc<Config>, Vec<InstanceContext>>>> =
    LazyLock::new(Mutex::default);

fn with_caller<T>(func: impl FnOnce(&mut Caller) -> T) -> T {
    CALLER.with(|caller| func(caller.borrow_mut().as_mut().unwrap()))
}

impl MemoryLike for WasmtimeMemory {
    fn fits_memory(&self, slice: MemSlice) -> Result<(), ()> {
        let end = slice.end::<usize>()?;
        if end <= with_caller(|caller| self.0.data_size(caller)) { Ok(()) } else { Err(()) }
    }

    fn view_memory(&self, slice: MemSlice) -> Result<Cow<[u8]>, ()> {
        let range = slice.range::<usize>()?;
        with_caller(|caller| {
            self.0.data(caller).get(range).map(|slice| Cow::Owned(slice.to_vec())).ok_or(())
        })
    }

    fn read_memory(&self, offset: u64, buffer: &mut [u8]) -> Result<(), ()> {
        let start = usize::try_from(offset).map_err(|_| ())?;
        let end = start.checked_add(buffer.len()).ok_or(())?;
        with_caller(|caller| {
            let memory = self.0.data(caller).get(start..end).ok_or(())?;
            buffer.copy_from_slice(memory);
            Ok(())
        })
    }

    fn write_memory(&mut self, offset: u64, buffer: &[u8]) -> Result<(), ()> {
        let start = usize::try_from(offset).map_err(|_| ())?;
        let end = start.checked_add(buffer.len()).ok_or(())?;
        with_caller(|caller| {
            let memory = self.0.data_mut(caller).get_mut(start..end).ok_or(())?;
            memory.copy_from_slice(buffer);
            Ok(())
        })
    }
}

trait IntoVMError {
    fn into_vm_error(self) -> Result<FunctionCallError, VMRunnerError>;
}

impl IntoVMError for anyhow::Error {
    fn into_vm_error(self) -> Result<FunctionCallError, VMRunnerError> {
        let cause = self.root_cause();
        if let Some(container) = cause.downcast_ref::<ErrorContainer>() {
            use {VMLogicError as LE, VMRunnerError as RE};
            return match container.take() {
                Some(LE::HostError(h)) => Ok(FunctionCallError::HostError(h)),
                Some(LE::ExternalError(s)) => Err(RE::ExternalError(s)),
                Some(LE::InconsistentStateError(e)) => Err(RE::InconsistentStateError(e)),
                None => panic!("error has already been taken out of the container?!"),
            };
        }
        if let Some(trap) = cause.downcast_ref::<wasmtime::Trap>() {
            use wasmtime::Trap as T;
            let nondeterministic_message = 'nondet: {
                return Ok(FunctionCallError::WasmTrap(match *trap {
                    T::StackOverflow => WasmTrap::StackOverflow,
                    T::MemoryOutOfBounds => WasmTrap::MemoryOutOfBounds,
                    T::TableOutOfBounds => WasmTrap::MemoryOutOfBounds,
                    T::IndirectCallToNull => WasmTrap::IndirectCallToNull,
                    T::BadSignature => WasmTrap::IncorrectCallIndirectSignature,
                    T::IntegerOverflow => WasmTrap::IllegalArithmetic,
                    T::IntegerDivisionByZero => WasmTrap::IllegalArithmetic,
                    T::BadConversionToInteger => WasmTrap::IllegalArithmetic,
                    T::UnreachableCodeReached => WasmTrap::Unreachable,
                    T::Interrupt => break 'nondet "interrupt",
                    T::HeapMisaligned => break 'nondet "heap misaligned",
                    t => {
                        return Err(VMRunnerError::WasmUnknownError {
                            debug_message: format!("unhandled trap type: {:?}", t),
                        });
                    }
                }));
            };
            return Err(VMRunnerError::Nondeterministic(nondeterministic_message.into()));
        }
        // FIXME: this can blow up in size and would get stored in the storage in case this was a
        // production runtime. Something more proper should be done here.
        Ok(FunctionCallError::LinkError { msg: format!("{:?}", cause) })
    }
}

pub(crate) fn default_wasmtime_config(c: &Config) -> wasmtime::Config {
    let features = crate::features::WasmFeatures::new(c);

    //let max_memory_size = usize::try_from(c.limit_config.max_memory_pages)
    //    .unwrap_or(usize::MAX)
    //    .saturating_mul(GUEST_PAGE_SIZE);
    //let mut pooling = PoolingAllocationConfig::default();
    //pooling
    //    .max_memory_size(max_memory_size)
    //    .table_elements(1_000_000)
    //    .total_component_instances(0)
    //    .total_core_instances(MAX_CONCURRENCY.into())
    //    .total_memories(MAX_CONCURRENCY.into())
    //    .total_tables(MAX_CONCURRENCY.into())
    //    // Minimize page faults on Linux
    //    .linear_memory_keep_resident(max_memory_size)
    //    .table_keep_resident(1_000_000 * size_of::<*const ()>());

    let mut config = wasmtime::Config::from(features);
    config
        // Enable copy-on-write heap images.
        .memory_init_cow(true)
        // wasm stack metering is implemented by instrumentation, we don't want wasmtime to trap before that
        .max_wasm_stack(1024 * 1024 * 1024)
        // enable the Cranelift optimizing compiler.
        .strategy(Strategy::Cranelift)
        // Enable signals-based traps. This is required to elide explicit bounds-checking.
        .signals_based_traps(true)
        // Configure linear memories such that explicit bounds-checking can be elided.
        .memory_reservation(1 << 32)
        .memory_guard_size(1 << 32);
    //.allocation_strategy(InstanceAllocationStrategy::Pooling(pooling));
    config
}

pub(crate) fn wasmtime_vm_hash() -> u64 {
    // TODO: take into account compiler and engine used to compile the contract.
    64
}

#[derive(Clone)]
pub(crate) struct WasmtimeVM {
    config: Arc<Config>,
    engine: Engine,
    //instances: ConcurrencySemaphore,
}

impl WasmtimeVM {
    pub(crate) fn new(config: Arc<Config>) -> Self {
        {
            if let Some(vm) = VMS.read().expect("failed to read-lock VM pool").get(&config) {
                return vm.clone();
            }
        }
        let engine = Engine::new(&default_wasmtime_config(&config))
            .expect("failed to contruct Wasmtime engine");
        //let instances = ConcurrencySemaphore::default();
        VMS.write()
            .expect("failed to write-lock VM pool")
            .entry(Arc::clone(&config))
            //.or_insert(Self { config, engine, instances })
            .or_insert(Self { config, engine })
            .clone()
    }

    #[tracing::instrument(target = "vm", level = "debug", "WasmtimeVM::compile_uncached", skip_all)]
    fn compile_uncached(&self, code: &ContractCode) -> Result<Vec<u8>, CompilationError> {
        let start = std::time::Instant::now();
        let prepared_code = prepare::prepare_contract(code.code(), &self.config, VMKind::Wasmtime)
            .map_err(CompilationError::PrepareError)?;
        let serialized = self.engine.precompile_module(&prepared_code).map_err(|err| {
            tracing::error!(?err, "wasmtime failed to compile the prepared code (this is defense-in-depth, the error was recovered from but should be reported to the developers)");
            CompilationError::WasmtimeCompileError { msg: err.to_string() }
        });
        crate::metrics::compilation_duration(VMKind::Wasmtime, start.elapsed());
        serialized
    }

    fn compile_and_cache(
        &self,
        code: &ContractCode,
        cache: &dyn ContractRuntimeCache,
    ) -> Result<Result<Vec<u8>, CompilationError>, CacheError> {
        let serialized_or_error = self.compile_uncached(code);
        let key = get_contract_cache_key(*code.hash(), &self.config);
        let record = CompiledContractInfo {
            wasm_bytes: code.code().len() as u64,
            compiled: match &serialized_or_error {
                Ok(serialized) => CompiledContract::Code(serialized.clone()),
                Err(err) => CompiledContract::CompileModuleError(err.clone()),
            },
        };
        cache.put(&key, record).map_err(CacheError::WriteError)?;
        Ok(serialized_or_error)
    }

    fn with_compiled_and_loaded(
        &self,
        cache: &dyn ContractRuntimeCache,
        contract: &dyn Contract,
        mut gas_counter: GasCounter,
        method: &str,
        closure: impl FnOnce(GasCounter, Module) -> VMResult<PreparedContract>,
    ) -> VMResult<PreparedContract> {
        type MemoryCacheType = (u64, Result<Module, CompilationError>);
        let to_any = |v: MemoryCacheType| -> Box<dyn std::any::Any + Send> { Box::new(v) };
        let key = get_contract_cache_key(contract.hash(), &self.config);
        let (wasm_bytes, module_result) = cache.memory_cache().try_lookup(
            key,
            || {
                let cache_record = cache.get(&key).map_err(CacheError::ReadError)?;
                let Some(compiled_contract_info) = cache_record else {
                    let Some(code) = contract.get_code() else {
                        return Err(VMRunnerError::ContractCodeNotPresent);
                    };
                    return Ok(to_any((
                        code.code().len() as u64,
                        match self.compile_and_cache(&code, cache)? {
                            Ok(serialized_module) => Ok(unsafe {
                                Module::deserialize(&self.engine, serialized_module)
                                    .map_err(|err| VMRunnerError::LoadingError(err.to_string()))?
                            }),
                            Err(err) => Err(err),
                        },
                    )));
                };
                match &compiled_contract_info.compiled {
                    CompiledContract::CompileModuleError(err) => Ok::<_, VMRunnerError>(to_any((
                        compiled_contract_info.wasm_bytes,
                        Err(err.clone()),
                    ))),
                    CompiledContract::Code(serialized_module) => {
                        unsafe {
                            // (UN-)SAFETY: the `serialized_module` must have been produced by
                            // a prior call to `serialize`.
                            //
                            // In practice this is not necessarily true. One could have
                            // forgotten to change the cache key when upgrading the version of
                            // the near_vm library or the database could have had its data
                            // corrupted while at rest.
                            //
                            // There should definitely be some validation in near_vm to ensure
                            // we load what we think we load.
                            let module = Module::deserialize(&self.engine, &serialized_module)
                                .map_err(|err| VMRunnerError::LoadingError(err.to_string()))?;
                            Ok(to_any((compiled_contract_info.wasm_bytes, Ok(module))))
                        }
                    }
                }
            },
            move |value| {
                let &(wasm_bytes, ref downcast) = value
                    .downcast_ref::<MemoryCacheType>()
                    .expect("downcast should always succeed");

                (wasm_bytes, downcast.clone())
            },
        )?;

        let config = Arc::clone(&self.config);
        let result = gas_counter.before_loading_executable(&config, &method, wasm_bytes);
        if let Err(e) = result {
            let result = PreparationResult::OutcomeAbort(e);
            return Ok(PreparedContract { config, gas_counter, result });
        }
        match module_result {
            Ok(module) => {
                let result = gas_counter.after_loading_executable(&config, wasm_bytes);
                if let Err(e) = result {
                    let result = PreparationResult::OutcomeAbort(e);
                    return Ok(PreparedContract { config, gas_counter, result });
                }
                closure(gas_counter, module)
            }
            Err(e) => {
                let result =
                    PreparationResult::OutcomeAbort(FunctionCallError::CompilationError(e));
                return Ok(PreparedContract { config, gas_counter, result });
            }
        }
    }
}

impl crate::runner::VM for WasmtimeVM {
    fn precompile(
        &self,
        code: &ContractCode,
        cache: &dyn ContractRuntimeCache,
    ) -> Result<
        Result<ContractPrecompilatonResult, CompilationError>,
        crate::logic::errors::CacheError,
    > {
        Ok(self
            .compile_and_cache(code, cache)?
            .map(|_| ContractPrecompilatonResult::ContractCompiled))
    }

    fn prepare(
        self: Box<Self>,
        code: &dyn Contract,
        cache: Option<&dyn ContractRuntimeCache>,
        gas_counter: GasCounter,
        method: &str,
    ) -> Box<dyn crate::PreparedContract> {
        let cache = cache.unwrap_or(&NoContractRuntimeCache);
        let prepd = self.with_compiled_and_loaded(
            cache,
            code,
            gas_counter,
            method,
            |gas_counter, module| {
                let config = Arc::clone(&self.config);
                let Some(Func(func_type)) = module.get_export(method) else {
                    let e =
                        FunctionCallError::MethodResolveError(MethodResolveError::MethodNotFound);
                    let result = PreparationResult::OutcomeAbortButNopInOldProtocol(e);
                    return Ok(PreparedContract { config, gas_counter, result });
                };
                if func_type.params().len() != 0 || func_type.results().len() != 0 {
                    let e = FunctionCallError::MethodResolveError(
                        MethodResolveError::MethodInvalidSignature,
                    );
                    let result = PreparationResult::OutcomeAbortButNopInOldProtocol(e);
                    return Ok(PreparedContract { config, gas_counter, result });
                }
                let Some(method) = module.get_export_index(method) else {
                    let e =
                        FunctionCallError::MethodResolveError(MethodResolveError::MethodNotFound);
                    let result = PreparationResult::OutcomeAbortButNopInOldProtocol(e);
                    return Ok(PreparedContract { config, gas_counter, result });
                };

                {
                    let mut cxs =
                        INSTANCE_CONTEXTS.lock().expect("failed to lock instance context pool");
                    let resources = module.resources_required();
                    let num_memories = resources.num_memories.try_into().unwrap_or(usize::MAX);
                    let num_tables = resources.num_tables.try_into().unwrap_or(usize::MAX);
                    if let Some(cxs) = cxs.get_mut(&self.config) {
                        while let Some(mut context) = cxs.pop() {
                            if let Some(budget) = context.memory_budget.checked_sub(num_memories) {
                                context.memory_budget = budget;
                            } else {
                                continue;
                            }
                            if let Some(budget) = context.table_budget.checked_sub(num_tables) {
                                context.table_budget = budget;
                            } else {
                                continue;
                            }
                            let result = PreparationResult::Ready(ReadyContract {
                                context,
                                module,
                                method: method.into(),
                            });
                            return Ok(PreparedContract { config, gas_counter, result });
                        }
                    }
                }
                let context = InstanceContext::new(module.engine(), &config.limit_config);
                let result = PreparationResult::Ready(ReadyContract { context, module, method });
                Ok(PreparedContract { config, gas_counter, result })
            },
        );
        Box::new(prepd)
    }
}

struct ReadyContract {
    context: InstanceContext,
    module: Module,
    method: ModuleExport,
    //instances: ConcurrencySemaphore,
}

struct PreparedContract {
    config: Arc<Config>,
    gas_counter: GasCounter,
    result: PreparationResult,
}

#[allow(clippy::large_enum_variant)]
enum PreparationResult {
    OutcomeAbortButNopInOldProtocol(FunctionCallError),
    OutcomeAbort(FunctionCallError),
    Ready(ReadyContract),
}

enum RunOutcome {
    Ok,
    AbortNop(FunctionCallError),
    Abort(FunctionCallError),
}

fn call(
    mut store: &mut Store<()>,
    instance: Instance,
    method: ModuleExport,
) -> Result<RunOutcome, VMRunnerError> {
    let Some(Extern::Func(func)) = instance.get_module_export(&mut store, &method) else {
        return Ok(RunOutcome::AbortNop(FunctionCallError::MethodResolveError(
            MethodResolveError::MethodNotFound,
        )));
    };
    match func.typed(&mut store) {
        Ok(run) => match run.call(store, ()) {
            Ok(()) => Ok(RunOutcome::Ok),
            Err(err) => err.into_vm_error().map(RunOutcome::Abort),
        },
        Err(err) => err.into_vm_error().map(RunOutcome::Abort),
    }
}

fn instantiate_and_call(
    mut store: &mut Store<()>,
    linker: &mut Linker<()>,
    module: &Module,
    method: ModuleExport,
) -> Result<RunOutcome, VMRunnerError> {
    match linker.instantiate(&mut store, module) {
        Ok(instance) => call(store, instance, method),
        Err(err) => err.into_vm_error().map(RunOutcome::Abort),
    }
}

impl crate::PreparedContract for VMResult<PreparedContract> {
    fn run(
        self: Box<Self>,
        ext: &mut dyn External,
        context: &VMContext,
        fees_config: Arc<RuntimeFeesConfig>,
    ) -> VMResult {
        let PreparedContract { config, gas_counter, result } = (*self)?;
        let result_state = ExecutionResultState::new(&context, gas_counter, config);
        let ReadyContract {
            context:
                InstanceContext { mut store, mut memory, instance_budget, table_budget, memory_budget },
            module,
            method,
            //instances,
        } = match result {
            PreparationResult::Ready(r) => r,
            PreparationResult::OutcomeAbortButNopInOldProtocol(e) => {
                return Ok(VMOutcome::abort_but_nop_outcome_in_old_protocol(result_state, e));
            }
            PreparationResult::OutcomeAbort(e) => {
                return Ok(VMOutcome::abort(result_state, e));
            }
        };
        let instance_budget = instance_budget.saturating_sub(1);

        let memory_copy = memory.0;
        let config = Arc::clone(&result_state.config);
        let mut logic = VMLogic::new(ext, context, fees_config, result_state, &mut memory);
        let mut linker = Linker::new(store.engine());
        // TODO: config could be accessed through `logic.result_state`, without this code having to
        // figure it out...
        link(&mut linker, memory_copy, &store, &config, &mut logic);

        let res = instantiate_and_call(&mut store, &mut linker, &module, method);
        if instance_budget > 0 && table_budget > 0 && memory_budget > 0 {
            let mut stores =
                INSTANCE_CONTEXTS.lock().expect("failed to lock instance context pool");
            stores.entry(config).or_default().push(InstanceContext {
                store,
                memory: WasmtimeMemory(memory_copy),
                instance_budget,
                table_budget,
                memory_budget,
            });
            lazy_drop(Box::new((linker, module)));
        } else {
            lazy_drop(Box::new((linker, module, store)));
        }
        match res? {
            RunOutcome::Ok => Ok(VMOutcome::ok(logic.result_state)),
            RunOutcome::AbortNop(error) => {
                Ok(VMOutcome::abort_but_nop_outcome_in_old_protocol(logic.result_state, error))
            }
            RunOutcome::Abort(error) => Ok(VMOutcome::abort(logic.result_state, error)),
        }
    }
}

/// This is a container from which an error can be taken out by value. This is necessary as
/// `anyhow` does not really give any opportunity to grab causes by value and the VM Logic
/// errors end up a couple layers deep in a causal chain.
#[derive(Debug)]
pub(crate) struct ErrorContainer(parking_lot::Mutex<Option<VMLogicError>>);
impl ErrorContainer {
    pub(crate) fn take(&self) -> Option<VMLogicError> {
        self.0.lock().take()
    }
}
impl std::error::Error for ErrorContainer {}
impl std::fmt::Display for ErrorContainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VMLogic error occurred and is now stored in an opaque storage container")
    }
}

thread_local! {
    static CALLER_CONTEXT: UnsafeCell<*mut c_void> = const { UnsafeCell::new(core::ptr::null_mut()) };
}

fn link<'a, 'b>(
    linker: &mut wasmtime::Linker<()>,
    memory: wasmtime::Memory,
    store: &wasmtime::Store<()>,
    config: &Config,
    logic: &'a mut VMLogic<'b>,
) {
    // Unfortunately, due to the Wasmtime implementation we have to do tricks with the
    // lifetimes of the logic instance and pass raw pointers here.
    // FIXME(nagisa): I believe this is no longer required, we just need to look at this code
    // again.
    let raw_logic = logic as *mut _ as *mut c_void;
    CALLER_CONTEXT.with(|caller_context| unsafe { *caller_context.get() = raw_logic });
    linker.define(store, "env", "memory", memory).expect("cannot define memory");

    macro_rules! add_import {
        (
          $mod:ident / $name:ident : $func:ident < [ $( $arg_name:ident : $arg_type:ident ),* ] -> [ $( $returns:ident ),* ] >
        ) => {
            #[allow(unused_parens)]
            fn $name(caller: wasmtime::Caller<'_, ()>, $( $arg_name: $arg_type ),* ) -> anyhow::Result<($( $returns ),*)> {
                const TRACE: bool = imports::should_trace_host_function(stringify!($name));
                let _span = TRACE.then(|| {
                    tracing::trace_span!(target: "vm::host_function", stringify!($name)).entered()
                });
                // the below is bad. don't do this at home. it probably works thanks to the exact way the system is setup.
                // Thankfully, this doesn't run in production, and hopefully should be possible to remove before we even
                // consider doing so.
                let data = CALLER_CONTEXT.with(|caller_context| {
                    unsafe {
                        *caller_context.get()
                    }
                });
                unsafe {
                    // Transmute the lifetime of caller so it's possible to put it in a thread-local.
                    #[allow(clippy::missing_transmute_annotations)]
                    crate::wasmtime_runner::CALLER.with(|runner_caller| *runner_caller.borrow_mut() = std::mem::transmute(caller));
                }
                let logic: &mut VMLogic<'_> = unsafe { &mut *(data as *mut VMLogic<'_>) };
                match logic.$func( $( $arg_name as $arg_type, )* ) {
                    Ok(result) => Ok(result as ($( $returns ),* ) ),
                    Err(err) => {
                        Err(ErrorContainer(parking_lot::Mutex::new(Some(err))).into())
                    }
                }
            }

            linker.func_wrap(stringify!($mod), stringify!($name), $name).expect("cannot link external");
        };
    }
    imports::for_each_available_import!(config, add_import);
}
