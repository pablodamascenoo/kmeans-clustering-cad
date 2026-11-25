#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <omp.h>

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

#pragma omp begin declare target
static inline unsigned int rand_device(unsigned int *seed)
{
  unsigned long long z = (unsigned long long)(*seed) + 0x9e3779b97f4a7c15ULL;
  *seed = (unsigned int)z;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return (unsigned int)((z ^ (z >> 31)) >> 32);
}
#pragma omp end declare target

void inicializar_centroides(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{
  int *escolhidos = malloc(num_clusters * sizeof(int));

  for (int i = 0; i < num_clusters; i++)
  {
    int random_index;
    int duplicado;
    do
    {
      duplicado = 0;
      random_index = rand() % num_pontos;
      for (int j = 0; j < i; j++)
      {
        if (escolhidos[j] == random_index)
        {
          duplicado = 1;
          break;
        }
      }
    } while (duplicado);

    escolhidos[i] = random_index;
    centroides[i].x = pontos[random_index].x;
    centroides[i].y = pontos[random_index].y;
  }

  free(escolhidos);
}

void atualizar_centroides(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos, double *soma_x, double *soma_y, int *count)
{
  #pragma omp target teams distribute parallel for
  for (int j = 0; j < num_clusters; j++)
  {
    if (count[j] > 0)
    {
      centroides[j].x = soma_x[j] / count[j];
      centroides[j].y = soma_y[j] / count[j];
    }
    else
    {
      unsigned int seed = (unsigned int)j + 1u;
      int random_index = (int)(rand_device(&seed) % (unsigned int)num_pontos);
      centroides[j].x = pontos[random_index].x;
      centroides[j].y = pontos[random_index].y;
    }
  }
}

int calcular_distancia_e_atribuir_clusters(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos, double *soma_x, double *soma_y, int *count)
{
  int mudancas = 0;


  #pragma omp target teams distribute parallel for \
    map(tofrom: mudancas) \
    reduction(+:mudancas, soma_x[0:num_clusters], soma_y[0:num_clusters], count[0:num_clusters])
  for (int i = 0; i < num_pontos; i++)
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
      mudancas++;
      pontos[i].cluster_id = cluster_id;
    }

    int id = pontos[i].cluster_id;
    if (id >= 0 && id < num_clusters)
    {
      soma_x[id] += pontos[i].x;
      soma_y[id] += pontos[i].y;
      count[id]++;
    }
  }

  atualizar_centroides(centroides, num_clusters, pontos, num_pontos, soma_x, soma_y, count);

  return mudancas;
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
  for (int i = 0; i < num_pontos; i++)
  {
    pontos[i].cluster_id = -1;
  }

  inicializar_centroides(centroides, num_clusters, pontos, num_pontos);

  double *soma_x = calloc(num_clusters, sizeof(double));
  double *soma_y = calloc(num_clusters, sizeof(double));
  int *count = calloc(num_clusters, sizeof(int));

  #pragma omp target enter data map(alloc: soma_x[0:num_clusters], soma_y[0:num_clusters], count[0:num_clusters])

  int iteracao;
  #pragma omp target data \
    map(tofrom: pontos[0:num_pontos], centroides[0:num_clusters])
  {
    for (iteracao = 0; iteracao < max_iteracoes; iteracao++)
    {
      #pragma omp target teams distribute parallel for
      for (int j = 0; j < num_clusters; j++)
      {
        soma_x[j] = 0.0;
        soma_y[j] = 0.0;
        count[j] = 0;
      }

      int mudancas = calcular_distancia_e_atribuir_clusters(
          centroides,
          num_clusters,
          pontos,
          num_pontos,
          soma_x,
          soma_y,
          count
        );
      if (mudancas == 0)
      {
        break;
      }
    }
  }

  #pragma omp target exit data map(delete: soma_x[0:num_clusters], soma_y[0:num_clusters], count[0:num_clusters])

  free(soma_x);
  free(soma_y);
  free(count);

  return iteracao;
}

// Executa o Método do Cotovelo e retorna o k ideal pela maior distância perpendicular à reta
int descobrir_k_ideal(Ponto *pontos, int num_pontos, int k_max, int max_iteracoes)
{
  if (k_max > num_pontos)
  {
    k_max = num_pontos;
  }

  if (k_max <= 2)
  {
    return k_max;
  }

  double *wcss_valores = malloc((k_max + 1) * sizeof(double));

  printf("\n=== BUSCA AUTOMATICA DO K IDEAL (METODO DO COTOVELO) ===\n");
  printf("%-10s | %-18s | %-15s | %-12s\n", "Clusters", "WCSS (Inercia)", "Reducao (%)", "Iteracoes");
  printf("-----------------------------------------------------------------\n");

  for (int k = 1; k <= k_max; k++)
  {
    Centroide *temp_centroides = malloc(k * sizeof(Centroide));
    int iters = executar_kmeans(temp_centroides, k, pontos, num_pontos, max_iteracoes);
    wcss_valores[k] = calcular_wcss(temp_centroides, k, pontos, num_pontos);

    if (k == 1)
    {
      printf("%-10d | %-18.6lf | %-15s | %-12d\n", k, wcss_valores[k], "-", iters);
    }
    else
    {
      double reducao = ((wcss_valores[k - 1] - wcss_valores[k]) / wcss_valores[k - 1]) * 100.0;
      printf("%-10d | %-18.6lf | %-14.2lf%% | %-12d\n", k, wcss_valores[k], reducao, iters);
    }

    free(temp_centroides);
  }

  // Normaliza os eixos X (k) e Y (WCSS) entre 0 e 1 para evitar distorção de escala na reta
  double x1 = 0.0, y1 = 1.0; // Ponto k = 1 (WCSS máximo)
  double x2 = 1.0, y2 = 0.0; // Ponto k = k_max (WCSS mínimo)

  double wcss_min = wcss_valores[k_max];
  double wcss_range = wcss_valores[1] - wcss_min;

  int k_ideal = 1;
  double max_distancia = -1.0;

  if (wcss_range > 0.0)
  {
    double dx_reta = x2 - x1;
    double dy_reta = y2 - y1;
    double comprimento_reta = sqrt((dx_reta * dx_reta) + (dy_reta * dy_reta));

    for (int k = 1; k <= k_max; k++)
    {
      double x0 = (double)(k - 1) / (k_max - 1);
      double y0 = (wcss_valores[k] - wcss_min) / wcss_range;

      // Distância perpendicular do ponto (x0, y0) até a reta entre (x1, y1) e (x2, y2)
      double numerador = fabs((dy_reta * x0) - (dx_reta * y0) + (x2 * y1) - (y2 * x1));
      double distancia = numerador / comprimento_reta;

      if (distancia > max_distancia)
      {
        max_distancia = distancia;
        k_ideal = k;
      }
    }
  }

  printf("=================================================================\n");
  printf("[LOG] Cotovelo detectado em k = %d clusters. Prosseguindo com execucao final...\n\n", k_ideal);

  free(wcss_valores);
  return k_ideal;
}

int leitura_entradas(int argc, char *argv[], int *num_clusters, Ponto **pontos, int *num_pontos, char **linha, FILE **arquivo)
{
  srand(time(NULL));

  if (argc != 4 && argc != 5)
  {
    printf("Uso: %s <arquivo_entrada> <coluna_x> <coluna_y> [<numero_clusters>]\n", argv[0]);
    return 0;
  }

  *num_clusters = -1;
  if (argc == 5)
  {
    *num_clusters = atoi(argv[4]);
    if (*num_clusters <= 0)
    {
      printf("Erro: O numero de clusters deve ser maior que zero.\n");
      return 0;
    }
  }

  *arquivo = fopen(argv[1], "r");
  if (*arquivo == NULL)
  {
    printf("Erro ao abrir o arquivo de entrada: %s\n", argv[1]);
    return 0;
  }

  const char *delim = ",";
  *linha = NULL;
  size_t len = 0;
  ssize_t nread;

  nread = getline(linha, &len, *arquivo);
  if (nread == -1)
  {
    printf("Erro ao ler o cabecalho do arquivo.\n");
    free(*linha);
    *linha = NULL;
    fclose(*arquivo);
    *arquivo = NULL;
    return 0;
  }

  (*linha)[strcspn(*linha, "\r\n")] = '\0';

  int index_coluna_x = -1;
  int index_coluna_y = -1;
  char *token = strtok(*linha, delim);
  int countIndex = 0;

  while (token != NULL)
  {
    if (strcmp(token, argv[2]) == 0)
    {
      index_coluna_x = countIndex;
    }
    else if (strcmp(token, argv[3]) == 0)
    {
      index_coluna_y = countIndex;
    }
    token = strtok(NULL, delim);
    countIndex++;
  }

  if (index_coluna_x == -1 || index_coluna_y == -1)
  {
    printf("Erro: Coluna '%s' ou '%s' nao encontrada no cabecalho.\n", argv[2], argv[3]);
    free(*linha);
    *linha = NULL;
    fclose(*arquivo);
    *arquivo = NULL;
    return 0;
  }

  printf("[LOG] Colunas mapeadas: '%s' (indice %d) e '%s' (indice %d)\n",
         argv[2], index_coluna_x, argv[3], index_coluna_y);

  int size = 1024;
  *pontos = malloc(size * sizeof(Ponto));
  if (*pontos == NULL)
  {
    printf("Erro: Falha ao alocar memoria para os pontos.\n");
    free(*linha);
    *linha = NULL;
    fclose(*arquivo);
    *arquivo = NULL;
    return 0;
  }
  *num_pontos = 0;
  int linhas_descartadas = 0;

  clock_t inicio_leitura = clock();

  while ((nread = getline(linha, &len, *arquivo)) != -1)
  {
    if (*num_pontos >= size)
    {
      size *= 2;
      Ponto *temp = realloc(*pontos, size * sizeof(Ponto));
      if (temp == NULL)
      {
        printf("Erro: Falha ao redimensionar o vetor de pontos.\n");
        free(*linha);
        *linha = NULL;
        free(*pontos);
        *pontos = NULL;
        fclose(*arquivo);
        *arquivo = NULL;
        return 0;
      }
      *pontos = temp;
    }

    (*linha)[strcspn(*linha, "\r\n")] = '\0';
    char *tok = strtok(*linha, delim);
    int col_atual = 0;
    double temp_x = 0.0, temp_y = 0.0;
    int achou_x = 0, achou_y = 0;

    while (tok != NULL)
    {
      if (col_atual == index_coluna_x)
      {
        temp_x = atof(tok);
        achou_x = 1;
      }
      else if (col_atual == index_coluna_y)
      {
        temp_y = atof(tok);
        achou_y = 1;
      }
      tok = strtok(NULL, delim);
      col_atual++;
    }

    if (achou_x && achou_y && temp_x != 0.0 && temp_y != 0.0)
    {
      (*pontos)[*num_pontos].x = temp_x;
      (*pontos)[*num_pontos].y = temp_y;
      (*pontos)[*num_pontos].cluster_id = -1;
      (*num_pontos)++;
    }
    else
    {
      linhas_descartadas++;
    }
  }

  double tempo_leitura = (double)(clock() - inicio_leitura) / CLOCKS_PER_SEC;
  printf("[LOG] Leitura concluida em %.3fs | Pontos validos: %d | Linhas descartadas: %d\n",
         tempo_leitura, *num_pontos, linhas_descartadas);

  if (*num_pontos == 0)
  {
    printf("Erro: Nenhum ponto valido encontrado no arquivo.\n");
    free(*linha);
    *linha = NULL;
    free(*pontos);
    *pontos = NULL;
    fclose(*arquivo);
    *arquivo = NULL;
    return 0;
  }

  // Se o número de clusters não foi informado (argc == 4), descobre via WCSS
  if (argc == 4)
  {
    *num_clusters = descobrir_k_ideal(*pontos, *num_pontos, 8, 100);
  }
  else if (*num_pontos < *num_clusters)
  {
    printf("Erro: Numero de pontos validos (%d) menor que o numero de clusters (%d).\n",
           *num_pontos, *num_clusters);
    free(*linha);
    *linha = NULL;
    free(*pontos);
    *pontos = NULL;
    fclose(*arquivo);
    *arquivo = NULL;
    return 0;
  }

  return 1;
}

int main(int argc, char *argv[])
{
  int num_clusters = -1;
  int num_pontos = 0;
  Ponto *pontos = NULL;
  char *linha = NULL;
  FILE *arquivo = NULL;

  if (!leitura_entradas(argc, argv, &num_clusters, &pontos, &num_pontos, &linha, &arquivo))
  {
    return EXIT_FAILURE;
  }

  Centroide *centroides = malloc(num_clusters * sizeof(Centroide));
  if (centroides == NULL)
  {
    printf("Erro: Falha ao alocar memoria para os centroides.\n");
    free(linha);
    free(pontos);
    fclose(arquivo);
    return EXIT_FAILURE;
  }

  // Executa o K-means definitivo
  clock_t inicio_kmeans = clock();

  int iteracoes_finais = executar_kmeans(centroides, num_clusters, pontos, num_pontos, 100);
  double tempo_kmeans = (double)(clock() - inicio_kmeans) / CLOCKS_PER_SEC;

  int *contagem_clusters = calloc(num_clusters, sizeof(int));
  for (int i = 0; i < num_pontos; i++)
  {
    int id = pontos[i].cluster_id;
    if (id >= 0 && id < num_clusters)
    {
      contagem_clusters[id]++;
    }
  }

  printf("=== RESULTADO FINAL (k = %d) ===\n", num_clusters);
  printf("[LOG] Convergiu em %d iteracoes (%.3fs)\n\n", iteracoes_finais, tempo_kmeans);

  for (int i = 0; i < num_clusters; i++)
  {
    double percentual = ((double)contagem_clusters[i] / num_pontos) * 100.0;
    printf("Centroide %d: x = %.13lf, y = %.13lf | Pontos: %d (%.2lf%%)\n",
           i, centroides[i].x, centroides[i].y, contagem_clusters[i], percentual);
  }

  free(contagem_clusters);
  free(linha);
  free(pontos);
  free(centroides);
  fclose(arquivo);

  return EXIT_SUCCESS;
}

// nvc -mp=gpu -gpu=cc89 -O3 -lm -o main_openmp main_openmp.c
// cc89 é um código que depende da GPU
