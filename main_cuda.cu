#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <cuda_runtime.h>

typedef struct
{
  double x;
  double y;
  int cluster_id;
} Ponto;

typedef struct
{
  double x;
  double y;
} Centroide;

#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
      fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)

__global__ void atribuir_clusters_kernel(Ponto *pontos, int num_pontos, Centroide *centroides, int num_clusters, int *d_mudancas)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < num_pontos)
  {
    double min_distancia = DBL_MAX;
    int cluster_id = -1;

    for (int j = 0; j < num_clusters; j++)
    {
      double dx = pontos[i].x - centroides[j].x;
      double dy = pontos[i].y - centroides[j].y;
      double distancia = (dx * dx) + (dy * dy);

      if (distancia < min_distancia)
      {
        min_distancia = distancia;
        cluster_id = j;
      }
    }

    if (pontos[i].cluster_id != cluster_id)
    {
      atomicAdd(d_mudancas, 1);
      pontos[i].cluster_id = cluster_id;
    }
  }
}

void inicializar_centroides(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{
  int *escolhidos = (int *)malloc(num_clusters * sizeof(int));
  for (int i = 0; i < num_clusters; i++)
  {
    int random_index, duplicado;
    do {
      duplicado = 0;
      random_index = rand() % num_pontos;
      for (int j = 0; j < i; j++) {
        if (escolhidos[j] == random_index) { duplicado = 1; break; }
      }
    } while (duplicado);

    escolhidos[i] = random_index;
    centroides[i].x = pontos[random_index].x;
    centroides[i].y = pontos[random_index].y;
  }
  free(escolhidos);
}

void atualizar_centroides(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{
  double *soma_x = (double *)calloc(num_clusters, sizeof(double));
  double *soma_y = (double *)calloc(num_clusters, sizeof(double));
  int *count = (int *)calloc(num_clusters, sizeof(int));

  for (int i = 0; i < num_pontos; i++)
  {
    int id = pontos[i].cluster_id;
    if (id >= 0 && id < num_clusters)
    {
      soma_x[id] += pontos[i].x;
      soma_y[id] += pontos[i].y;
      count[id]++;
    }
  }

  for (int j = 0; j < num_clusters; j++)
  {
    if (count[j] > 0)
    {
      centroides[j].x = soma_x[j] / count[j];
      centroides[j].y = soma_y[j] / count[j];
    }
    else
    {
      int random_index = rand() % num_pontos;
      centroides[j].x = pontos[random_index].x;
      centroides[j].y = pontos[random_index].y;
    }
  }

  free(soma_x); free(soma_y); free(count);
}

double calcular_wcss(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{
  double wcss = 0.0;
  for (int i = 0; i < num_pontos; i++)
  {
    int id = pontos[i].cluster_id;
    if (id >= 0 && id < num_clusters)
    {
      double dx = pontos[i].x - centroides[id].x;
      double dy = pontos[i].y - centroides[id].y;
      wcss += (dx * dx) + (dy * dy);
    }
  }
  return wcss;
}

int executar_kmeans(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos, int max_iteracoes)
{
  for (int i = 0; i < num_pontos; i++) pontos[i].cluster_id = -1;
  inicializar_centroides(centroides, num_clusters, pontos, num_pontos);

  // Alocação de memória na GPU
  Ponto *d_pontos;
  Centroide *d_centroides;
  int *d_mudancas;

  CUDA_CHECK(cudaMalloc((void **)&d_pontos, num_pontos * sizeof(Ponto)));
  CUDA_CHECK(cudaMalloc((void **)&d_centroides, num_clusters * sizeof(Centroide)));
  CUDA_CHECK(cudaMalloc((void **)&d_mudancas, sizeof(int)));

  CUDA_CHECK(cudaMemcpy(d_pontos, pontos, num_pontos * sizeof(Ponto), cudaMemcpyHostToDevice));

  int block_size = 256;
  int grid_size = (num_pontos + block_size - 1) / block_size;

  int iteracao;
  for (iteracao = 0; iteracao < max_iteracoes; iteracao++)
  {
    int mudancas = 0;
    CUDA_CHECK(cudaMemcpy(d_mudancas, &mudancas, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroides, centroides, num_clusters * sizeof(Centroide), cudaMemcpyHostToDevice));

    atribuir_clusters_kernel<<<grid_size, block_size>>>(d_pontos, num_pontos, d_centroides, num_clusters, d_mudancas);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&mudancas, d_mudancas, sizeof(int), cudaMemcpyDeviceToHost));

    if (mudancas == 0)
    {
      CUDA_CHECK(cudaMemcpy(pontos, d_pontos, num_pontos * sizeof(Ponto), cudaMemcpyDeviceToHost));
      break;
    }

    CUDA_CHECK(cudaMemcpy(pontos, d_pontos, num_pontos * sizeof(Ponto), cudaMemcpyDeviceToHost));
    atualizar_centroides(centroides, num_clusters, pontos, num_pontos);
  }

  CUDA_CHECK(cudaFree(d_pontos));
  CUDA_CHECK(cudaFree(d_centroides));
  CUDA_CHECK(cudaFree(d_mudancas));

  return iteracao;
}

int descobrir_k_ideal(Ponto *pontos, int num_pontos, int k_max, int max_iteracoes)
{
  if (k_max > num_pontos) k_max = num_pontos;
  if (k_max <= 2) return k_max;

  double *wcss_valores = (double *)malloc((k_max + 1) * sizeof(double));

  printf("\n=== BUSCA AUTOMATICA DO K IDEAL (METODO DO COTOVELO - CUDA) ===\n");
  printf("%-10s | %-18s | %-15s | %-12s\n", "Clusters", "WCSS (Inercia)", "Reducao (%)", "Iteracoes");
  printf("-----------------------------------------------------------------\n");

  for (int k = 1; k <= k_max; k++)
  {
    Centroide *temp_centroides = (Centroide *)malloc(k * sizeof(Centroide));
    int iters = executar_kmeans(temp_centroides, k, pontos, num_pontos, max_iteracoes);
    wcss_valores[k] = calcular_wcss(temp_centroides, k, pontos, num_pontos);

    if (k == 1) printf("%-10d | %-18.6lf | %-15s | %-12d\n", k, wcss_valores[k], "-", iters);
    else printf("%-10d | %-18.6lf | %-14.2lf%% | %-12d\n", k, wcss_valores[k], ((wcss_valores[k - 1] - wcss_valores[k]) / wcss_valores[k - 1]) * 100.0, iters);

    free(temp_centroides);
  }

  double x1 = 0.0, y1 = 1.0, x2 = 1.0, y2 = 0.0;
  double wcss_min = wcss_valores[k_max];
  double wcss_range = wcss_valores[1] - wcss_min;
  int k_ideal = 1;
  double max_distancia = -1.0;

  if (wcss_range > 0.0)
  {
    double dx_reta = x2 - x1, dy_reta = y2 - y1;
    double comprimento_reta = sqrt((dx_reta * dx_reta) + (dy_reta * dy_reta));

    for (int k = 1; k <= k_max; k++)
    {
      double x0 = (double)(k - 1) / (k_max - 1);
      double y0 = (wcss_valores[k] - wcss_min) / wcss_range;
      double distancia = fabs((dy_reta * x0) - (dx_reta * y0) + (x2 * y1) - (y2 * x1)) / comprimento_reta;

      if (distancia > max_distancia) { max_distancia = distancia; k_ideal = k; }
    }
  }

  printf("=================================================================\n");
  printf("[LOG] Cotovelo detectado em k = %d clusters. Prosseguindo...\n\n", k_ideal);
  free(wcss_valores);
  return k_ideal;
}

int main(int argc, char *argv[])
{
  srand(time(NULL));

  if (argc != 4 && argc != 5)
  {
    printf("Uso: %s <arquivo_entrada> <coluna_x> <coluna_y> [<numero_clusters>]\n", argv[0]);
    return EXIT_FAILURE;
  }

  int num_clusters = -1;
  if (argc == 5) num_clusters = atoi(argv[4]);

  FILE *arquivo_entrada = fopen(argv[1], "r");
  if (arquivo_entrada == NULL) { printf("Erro ao abrir arquivo: %s\n", argv[1]); return EXIT_FAILURE; }

  const char *delim = ",";
  char *linha = NULL;
  size_t len = 0;
  ssize_t nread = getline(&linha, &len, arquivo_entrada);
  linha[strcspn(linha, "\r\n")] = '\0';

  int index_coluna_x = -1, index_coluna_y = -1, countIndex = 0;
  char *token = strtok(linha, delim);
  while (token != NULL)
  {
    if (strcmp(token, argv[2]) == 0) index_coluna_x = countIndex;
    else if (strcmp(token, argv[3]) == 0) index_coluna_y = countIndex;
    token = strtok(NULL, delim);
    countIndex++;
  }

  int size = 1024, num_pontos = 0, linhas_descartadas = 0;
  Ponto *pontos = (Ponto *)malloc(size * sizeof(Ponto));

  clock_t inicio_leitura = clock();
  while ((nread = getline(&linha, &len, arquivo_entrada)) != -1)
  {
    if (num_pontos >= size) { size *= 2; pontos = (Ponto *)realloc(pontos, size * sizeof(Ponto)); }

    linha[strcspn(linha, "\r\n")] = '\0';
    char *tok = strtok(linha, delim);
    int col_atual = 0, achou_x = 0, achou_y = 0;
    double temp_x = 0.0, temp_y = 0.0;

    while (tok != NULL)
    {
      if (col_atual == index_coluna_x) { temp_x = atof(tok); achou_x = 1; }
      else if (col_atual == index_coluna_y) { temp_y = atof(tok); achou_y = 1; }
      tok = strtok(NULL, delim);
      col_atual++;
    }

    if (achou_x && achou_y && temp_x != 0.0 && temp_y != 0.0)
    {
      pontos[num_pontos].x = temp_x;
      pontos[num_pontos].y = temp_y;
      pontos[num_pontos].cluster_id = -1;
      num_pontos++;
    }
    else linhas_descartadas++;
  }

  printf("[LOG] Leitura em %.3fs | Pontos: %d | Descartados: %d\n",
         (double)(clock() - inicio_leitura) / CLOCKS_PER_SEC, num_pontos, linhas_descartadas);

  if (argc == 4) num_clusters = descobrir_k_ideal(pontos, num_pontos, 8, 100);

  Centroide *centroides = (Centroide *)malloc(num_clusters * sizeof(Centroide));
  clock_t inicio_kmeans = clock();
  int iteracoes_finais = executar_kmeans(centroides, num_clusters, pontos, num_pontos, 100);
  double tempo_kmeans = (double)(clock() - inicio_kmeans) / CLOCKS_PER_SEC;

  int *contagem_clusters = (int *)calloc(num_clusters, sizeof(int));
  for (int i = 0; i < num_pontos; i++) contagem_clusters[pontos[i].cluster_id]++;

  printf("\n=== RESULTADO FINAL (k = %d - CUDA) ===\n", num_clusters);
  printf("[LOG] Convergiu em %d iteracoes (%.3fs)\n\n", iteracoes_finais, tempo_kmeans);

  for (int i = 0; i < num_clusters; i++)
  {
    double percentual = ((double)contagem_clusters[i] / num_pontos) * 100.0;
    printf("Centroide %d: x = %.13lf, y = %.13lf | Pontos: %d (%.2lf%%)\n",
           i, centroides[i].x, centroides[i].y, contagem_clusters[i], percentual);
  }

  free(contagem_clusters); free(linha); free(pontos); free(centroides); fclose(arquivo_entrada);
  return EXIT_SUCCESS;
}


/* .slurm

 #!/bin/bash
 #SBATCH --job-name=cuda
 #SBATCH --output=cuda_%j.out
 #SBATCH --error=cuda_%j.err
 #SBATCH --partition=gpu-4-a100
 #SBATCH --nodes=1
 #SBATCH --ntasks=1
 #SBATCH --gpus=1
 #SBATCH --time=00:10:00

 module purge
 module load compilers/nvidia/nvhpc/24.11

 nvcc -O3 main_cuda.cu -o main_cuda -lm

 ./main_cuda yellow_tripdata_2015-01.csv pickup_longitude pickup_latitude

 */
