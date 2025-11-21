# Parcial Corte 3 Paradigmas de Programacion

---

## Prerrequisitos

* **Sistema**: Linux (probado en distribuciones modernas).
* **Python**: Python 3.8+ instalado.
* **Rust**: `rustc`/`cargo` si se desea compilar con Cargo.

## Instalación del entorno

Se recomienda crear un entorno virtual para Python e instalar dependencias:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

(O altern.)

```bash
pip install -m requirements.txt
```

## Cómo ejecutar los ejemplos

* Ejecutar Python (Punto 3 - benchmark):

```bash
python3 punto3/bench_py.py
```

* Compilar y ejecutar el ejemplo en Rust (archivo único `linear.rs`):

```bash
rustc -O punto3/linear.rs -o punto3/linear
./punto3/linear
```

---

# Punto 1 — Diseño: paradigma de concurrencia + cálculo de π

**Objetivo**: entrenar una regresión lineal (GD) usando concurrencia para acelerar cálculo de gradientes/IO y simultáneamente calcular π en background sin bloquear.

### Componentes

* **Coordinator / ParameterServer**: mantiene parámetros `w,b`, `lr`, `lock`; expone `get_params()`, `apply_gradients(dw,db)`.
* **Worker [1..n]**: tiene `X_chunk,y_chunk`; métodos `compute_local_gradient(w,b)`, `push_gradient(dw,db)`.
* **GradientAggregator**: recibe gradientes de workers y realiza reducción (suma/avg).
* **DataLoader**: divide dataset en shards/mini-batches y sirve a workers.
* **PIWorker(s)**: ejecutan cálculo de π (Monte Carlo o Leibniz) de forma continua y publican `pi_estimate` en una estructura compartida (lock-free lectura preferida).
* **Logger / Monitor**: registra MSE, `w,b`, y `pi_estimate` periódicamente (cola no bloqueante).

### Sincronización y estrategias

* **Síncrono (recomendado)**: por mini-batch: broadcast `w,b` → workers calculan → envío de gradientes → agregación → actualización atómica en Coordinator → barrier → siguiente batch.
* **Asíncrono (opcional)**: workers actualizan PS sin barrier (Hogwild); más throughput, posible inestabilidad.
* **Comunicación**: colas/IPC/shared memory/RPC. En Python usar `multiprocessing` para evitar GIL en CPU-bound.

### Flujo (por epoch)

1. Coordinator envía `w,b` a Workers.
2. Workers calculan `dw_i,db_i` en paralelo y push a Aggregator.
3. Aggregator reduce → Coordinator aplica `apply_gradients()` (lock alrededor de la actualización si es necesario).
4. Logger registra métricas y lee `pi_estimate` de PIWorker.
5. PIWorker corre continuamente y publica estimados sin sincronizar con barrier.

**Diagrama (concurrencia + PI)**

* Imagen (diagrama de concurrencia + cálculo de π):
  ![Diagrama Punto 1](punto1/Diagrama_PP_concurrencia_y_calculo_PI.png)

---

# Punto 2 — Diseño: paradigma de Aspectos (AOP)

**Idea**: encapsular cross-cutting concerns (logging, sincronización, particionado, agregación de gradientes, orquestación PI) como *aspects* que se tejen sobre el núcleo (Trainer, Optimizer, DataLoader, Model).

### Núcleo (Core)

* `LinearModel(w,b)` — `predict(x)`, `loss(y,yhat)`.
* `Optimizer(lr)` — `compute_gradients(model, X_batch, y_batch)`.
* `Trainer(epochs,batch_size)` — `train()`, `apply_gradients(dw,db)`, `endEpoch()`.
* `DataLoader(X,y,shards)` — `get_shard(i)`.
* `PIService(total_samples)` — `compute_chunk(n)`, `get_estimate()`.

### Aspects (principales)

* **LoggingAspect**: pointcut `endEpoch()` → `after` -> log metrics (mse,w,b,pi).
* **SyncAspect**: pointcut `applyGradients()` → `around` -> acquire lock, proceed, release lock.
* **ShardAspect**: pointcut `computeGradients()` → `before` -> asignar shard y seed de RNG.
* **GradientAggregationAspect**: pointcut `pushGradient()` → `around` -> agregar/reducir gradientes antes de proceed.
* **PIAspect**: pointcut `trainerStart()` → `around` -> arrancar PIService en background; `after periodic()` -> leer `get_estimate()` y publicar.
* **EarlyStopAspect** (opcional): `after endEpoch()` -> evaluar convergencia y abortar si corresponde.

### Join points y advices (ejemplos)

* `execution(Trainer.train())` — `around` (PIAspect arranca/stops PIService).
* `call(Optimizer.compute_gradients(..))` — `before` (ShardAspect) y `after` (instrumentación).
* `call(Trainer.apply_gradients(dw,db))` — `around` (SyncAspect + GradientAggregation).
* `execution(Trainer.endEpoch())` — `after` (LoggingAspect, EarlyStopAspect).

**Diagrama (AOP)**

* Imagen (diagrama AOP / aspectos):
  ![Diagrama Punto 2](punto2/Diagrama_POA.png)

---

# Punto 3 — Implementación en Rust y comparación con Python (resumen y análisis)

## Código usados (resumen)

* **Python**: `punto3/bench_py.py` (GD puro con numpy vectorizado). Mide tiempo con `time.perf_counter()`, memoria con `tracemalloc` y `resource.getrusage()`.
* **Rust**: `punto3/linear.rs` (GD puro con loops explícitos). Mide tiempo con `Instant::now()` y RSS leyendo `/proc/self/status` (VmRSS).

## Resultados obtenidos

* **Python**

  * `time_s`: **0.044686 s**
  * `tracemalloc_peak_kb`: **1 KB**
  * `ru_maxrss_kb`: **52672 KB**

* **Rust**

  * `time_s`: **0.002050 s**
  * `rss_kb`: **2228 KB**

## Interpretación concisa

* **Tiempo**: Rust es ~22× más rápido en este microbenchmark. Explicación: Rust compila a código nativo optimizado; el bucle numérico está en código máquina. Python tiene sobrecosto del intérprete y gestión de arrays/objetos, aun cuando numpy delega operaciones a C; en su caso la implementación hace uso de numpy pero sigue habiendo overhead en la llamada/loop.
* **Memoria**: `tracemalloc` mide solo asignaciones rastreadas por Python (muy pequeñas). `ru_maxrss` reporta el RSS completo del proceso — incluye intérprete, librerías, arenas de allocator. Rust produce un binario pequeño sin VM y con menor RSS.

## Explicación breve del código

* **Python** (`bench_py.py`)

  * Define `train_linear()` que implementa descenso de gradiente batch completo: calcula predicción vectorizada `y_pred = w*X + b`, error, gradientes `dw, db`, actualiza `w,b`.
  * Mide tiempo con `perf_counter()`; memoria con `tracemalloc` y `getrusage`.

* **Rust** (`linear.rs`)

  * Define vectores `x,y`, variables `w,b`, y un loop por `epochs` con acumuladores `dw,db` calculados manualmente (loop sobre índices).
  * Mide tiempo con `Instant::now()` y lee `/proc/self/status` para `VmRSS`.

---