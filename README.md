# K-Means Clustering - NYC Yellow Taxi Trip Data (CAD)

Este repositório contém 4 implementações do algoritmo de clusterização **K-Means** voltadas para o estudo de **Computação de Alto Desempenho (CAD)**, utilizando dados geográficos das corridas dos táxis amarelos de Nova York (*NYC Yellow Taxi Trip Data*).

O objetivo do projeto é comparar o desempenho, aceleração (*speedup*) e escalabilidade entre a versão sequencial e diferentes abordagens de paralelização (OpenMP, MPI e CUDA).

---

## 📁 Estrutura do Repositório

* `main.c` — Implementação Sequencial em C puro.
* `main_openmp.c` — Implementação Paralela com OpenMP (Threads de CPU / Offloading).
* `main_mpi.c` — Implementação Paralela Distribuída com MPI (Memória Distribuída).
* `main_cuda.cu` — Implementação Paralela com CUDA (Aceleração por GPU).

---

## 📊 Dataset

O algoritmo utiliza o dataset **NYC Yellow Taxi Trip Data** (~1.99 GB).

* **Arquivo**: `yellow_tripdata_2015-01.csv`(Janeiro de 2015).
* **Colunas utilizadas**: `pickup_longitude` (coluna X) e `pickup_latitude` (coluna Y).

---

## Compilação e execução

### 1. Versão Sequencial (`main.c`)
Ambiente: Qualquer computador ou CPU local.

```bash
# Compilação
gcc -O3 main.c -o main -lm

# Execução (com k = 3 fixo)
./main yellow_tripdata_2015-01.csv Pickup_longitude Pickup_latitude 3

# Execução com busca automática do k ideal (Método do Cotovelo)
./main yellow_tripdata_2015-01.csv pickup_longitude pickup_latitude
```

### 2. Versão OpenMP (`main_openmp.c`)
Ambiente: CPU Multi-core local ou nó de cluster.

```bash
# Compilação para CPU
gcc -O3 -fopenmp main_openmp.c -o main_openmp -lm

# Definição do número de threads
export OMP_NUM_THREADS=4

# Execução
./main_openmp yellow_tripdata_2015-01.csv pickup_longitude pickup_latitude 3
```

### 3. Versão MPI (`main_mpi.c`)
Ambiente: Múltiplos núcleos locais ou múltiplos nós em cluster.

```bash
# Compilação com wrapper MPI
mpicc -O3 main_mpi.c -o main_mpi -lm

# Execução local (exemplo com 4 processos)
mpirun -np 4 ./main_mpi yellow_tripdata_2015-01.csv pickup_longitude pickup_latitude 3
```

### 4. Versão CUDA (`main_cuda.cu`)
Ambiente: Nó de execução com GPU NVIDIA (ex: NPAD/UFRN com GPU A100).

```bash
# Compilação manual com NVCC
nvcc -O3 main_cuda.cu -o main_cuda -lm

# Execução manual no nó com GPU
./main_cuda yellow_tripdata_2015-01.csv pickup_longitude pickup_latitude 3

# Submissão no cluster NPAD via Slurm
sbatch cuda.slurm
```

---

## ⚙️ Método do Cotovelo (Elbow Method)
Se o argumento opcional de número de clusters `[<numero_clusters>]` for omitido na execução, o programa roda automaticamente a busca pelo **k ideal** variando \\(k\\) de 1 a 8, calculando a inércia (WCSS) e selecionando o ponto de maior distância perpendicular à reta de referência.
