#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>

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

void inicializar_centroides(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{
  for (int i = 0; i < num_clusters; i++)
  {
    int random_index = rand() % num_pontos;
    centroides[i].x = pontos[random_index].x;
    centroides[i].y = pontos[random_index].y;
  }
}

int calcular_distancia_e_atribuir_clusters(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{

  int mudancas = 0;
  for (int i = 0; i < num_pontos; i++)
  {
    double min_distancia = DBL_MAX;
    int cluster_id = -1;

    for (int j = 0; j < num_clusters; j++)
    {
      double dx = pontos[i].x - centroides[j].x;
      double dy = pontos[i].y - centroides[j].y;
      double distancia = (dx) * (dx) + (dy) * (dy);

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
  }

  return mudancas;
}

void atualizar_centroides(Centroide *centroides, int num_clusters, Ponto *pontos, int num_pontos)
{
  for (int i = 0; i < num_clusters; i++)
  {
    double soma_x = 0.0;
    double soma_y = 0.0;
    int count = 0;

    for (int j = 0; j < num_pontos; j++)
    {
      if (pontos[j].cluster_id == i)
      {
        soma_x += pontos[j].x;
        soma_y += pontos[j].y;
        count++;
      }
    }

    if (count > 0)
    {
      centroides[i].x = soma_x / count;
      centroides[i].y = soma_y / count;
    }
  }
}

int main(int argc, char *argv[])
{
  srand(time(NULL));

  if (argc != 5)
  {
    printf("Devem ser passados os argumentos: <arquivo_entrada> <coluna_x> <coluna_y> <numero_clusters>\n");
    return EXIT_FAILURE;
  }

  int num_clusters = atoi(argv[4]);
  int index_coluna_x = -1;
  int index_coluna_y = -1;

  FILE *arquivo_entrada = fopen(argv[1], "r");
  if (arquivo_entrada == NULL)
  {
    printf("Erro ao abrir o arquivo de entrada.\n");
    return EXIT_FAILURE;
  }

  const char *delim = ",";

  char *linha = NULL;
  size_t len = 0;
  ssize_t nread;

  nread = getline(&linha, &len, arquivo_entrada);

  if (nread == -1)
  {
    printf("Erro ao ler a primeira linha do arquivo.\n");
    free(linha);

    fclose(arquivo_entrada);
    return EXIT_FAILURE;
  }

  linha[strcspn(linha, "\r\n")] = '\0';

  char *token = strtok(linha, delim);

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
    printf("Erro: Coluna_x ou Coluna_y não encontrada no arquivo de entrada.\n");
    free(linha);

    fclose(arquivo_entrada);
    return EXIT_FAILURE;
  }

  int size = 1024;

  Ponto *pontos = malloc(size * sizeof(Ponto));
  Centroide *centroides = malloc(num_clusters * sizeof(Centroide));

  int num_pontos = 0;
  while ((nread = getline(&linha, &len, arquivo_entrada)) != -1)
  {
    countIndex = 0;
    if (num_pontos >= size)
    {
      size *= 2;
      pontos = realloc(pontos, size * sizeof(Ponto));
    }

    linha[strcspn(linha, "\r\n")] = '\0';
    char *token = strtok(linha, delim);
    while (token != NULL)
    {
      if ((countIndex == index_coluna_x || countIndex == index_coluna_y) && (atof(token) == 0.0))
      {
        token = NULL;
        num_pontos--;
        continue;
      }
      if (countIndex == index_coluna_x)
      {
        pontos[num_pontos].x = atof(token);
      }
      else if (countIndex == index_coluna_y)
      {
        pontos[num_pontos].y = atof(token);
      }
      token = strtok(NULL, delim);
      countIndex++;
    }
    num_pontos++;
  }

  inicializar_centroides(centroides, num_clusters, pontos, num_pontos);

  printf("Primeira definição dos centroides:\n");
  for (int i = 0; i < num_clusters; i++)
  {
    printf("Centroide %d: x = %.13lf, y = %.13lf\n", i, centroides[i].x, centroides[i].y);
  }

  int iteracao;

  for (iteracao = 0; iteracao < 100; iteracao++)
  {
    int mudancas = calcular_distancia_e_atribuir_clusters(centroides, num_clusters, pontos, num_pontos);
    if (mudancas == 0)
    {
      printf("Nenhuma mudança detectada. Finalizando na iteração %d.\n", iteracao);
      break;
    }
    atualizar_centroides(centroides, num_clusters, pontos, num_pontos);
  }

  printf("\nCentroides finais após %d iterações:\n", iteracao);

  for (int i = 0; i < num_clusters; i++)
  {
    printf("Centroide %d: x = %.13lf, y = %.13lf\n", i, centroides[i].x, centroides[i].y);
  }

  free(linha);
  free(pontos);
  free(centroides);
  fclose(arquivo_entrada);

  return EXIT_SUCCESS;
}
