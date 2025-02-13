# CUDA-Project-Course
Projekt na kurs, omówienie problemu.

# Obliczenia iloczynu macierzy (kwadratowych) na procesorach karty graficznej

# Cel projektu

Celem laboratorium było zbadanie, czy zwiększenie ilości pracy realizowanej przez
pojedynczy wątek, polegające na obliczaniu większej liczby wyników w każdym
kroku, pozwoli na wzrost efektywności obliczeń. Standardowa implementacja
mnożenia macierzy na GPU (CUDA-NVIDIA) przydziela każdemu wątkowi obliczenie
jednego elementu macierzy wynikowej C. W projekcie sprawdzono możliwość
zmodyfikowania tego podejścia poprzez przydzielenie wątkom większej liczby
elementów do obliczenia.

GPU: NVIDIA GeForce GTX 1080

![image](https://github.com/user-attachments/assets/ba86fcfd-625a-466a-91d6-badb327a05e0)

# Analiza problemu

Naszym celem jest usprawnienie algorytmu mnożenia macierzy. Proces ten polega na
obliczaniu macierzy C, która jest wynikiem iloczynu macierzy A i B. Każdy element macierzy
C jest sumą iloczynów odpowiednich elementów z wiersza macierzy A i kolumny macierzy B.

Przykładowy kod NVIDIA, zawarty w dokumentacji CUDA, stosuje pamięć współdzieloną w
celu efektywnego ponownego wykorzystania danych. Mnożenie macierzy jest oparte na
podejściu kafelkowym, a dodatkowo zaimplementowano scalony dostęp do pamięci
globalnej.
```c++
// Indeks bloku
 int bx = blockIdx.x;
 int by = blockIdx.y;
 // Indeks wątku w obrębie bloku
 int tx = threadIdx.x;
 int ty = threadIdx.y;
 // Indeks pierwszej submacierzy macierzy A przetwarzanej przez blok
 int aBegin = wA * BLOCK_SIZE * by;
 // Indeks ostatniej submacierzy macierzy A przetwarzanej przez blok
 int aEnd = aBegin + wA - 1;
 // Wielkość kroku do iteracji przez submacierze A
 int aStep = BLOCK_SIZE;
 // Indeks pierwszej submacierzy macierzy B przetwarzanej przez blok
 int bBegin = BLOCK_SIZE * bx;
 // Wielkość kroku do iteracji przez submacierze B
 int bStep = BLOCK_SIZE * wB;
 // Zmienna przechowująca obliczany element submacierzy bloku
 float Csub = 0;
// Pętla przechodząca przez wszystkie submacierze A i B
 // wymagane do obliczenia submacierzy bloku
 for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
// Deklaracja tablicy w pamięci współdzielonej dla submacierzy A i B
         __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
         __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
//Załadowanie submacierzy A i B z pamięci globalnej do pamięci współdzielonej, każdy wątek ładuje jeden element z każdej macierzy
         As[ty][tx] = A[a + wA * ty + tx];
         Bs[ty][tx] = B[b + wB * ty + tx];
// Synchronizacja wątków, aby upewnić się, że wszystkie dane zostały załadowane
         __syncthreads();
 // Mnożenie submacierzy; każdy wątek oblicza jeden element
 // submacierzy wynikowej C
#pragma unroll
 for (int k = 0; k < BLOCK_SIZE; ++k) {
         Csub += As[ty][k] * Bs[k][tx];
 }
 // Synchronizacja wątków, aby upewnić się, że poprzednie
// obliczenia zostały zakończone przed załadowaniem nowych submacierzy
         __syncthreads();
 }
// Zapis submacierzy wyniku do pamięci globalnej, każdy wątek zapisuje jeden element
 int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
 C[c + wB * ty + tx] = Csub;
}
```
Kod wykorzystuje pamięć współdzieloną do optymalizacji dostępu do danych. Pamięć
współdzielona przechowuje fragmenty macierzy A i B (tablice As i Bs zadeklarowane jako
__shared__). Umożliwia to jednokrotne ładowanie często wykorzystywanych danych z
pamięci globalnej, co znacznie przyspiesza obliczenia, ponieważ wątki w bloku mogą
współdzielić te dane.

Każdy wątek oblicza jeden element macierzy wynikowej C. Dane z pamięci globalnej są
ładowane w sposób zoptymalizowany: kolejne wątki w bloku uzyskują dostęp do sąsiednich
adresów pamięci, co maksymalizuje przepustowość.

Wzorzec dostępu do macierzy A to A[a + width * ty + tx], a do macierzy B to
B[b + wB * ty + tx]. Wyniki są również zapisywane w sposób scalony:
C[c + wB * ty + tx].

Algorytm dzieli macierze A i B na podmacierze o rozmiarze BLOCK_SIZE × BLOCK_SIZE.
Bloki wątków przetwarzają poszczególne kafelki macierzy wynikowej C. W każdej iteracji
zewnętrznej pętli kafelki A i B są ładowane do pamięci współdzielonej, mnożone przez
siebie, a wyniki akumulowane w zmiennej lokalnej wątku. Synchronizacja wątków za
pomocą __syncthreads zapewnia poprawne współdzielenie danych między wątkami w
bloku. Na koniec każdy wątek zapisuje wynik w odpowiednie miejsce w macierzy C.

# Zwiększenie nakładu pracy na wątek

Wzrost liczby wyników częściowych obliczanych, w każdym z opisanych powyżej etapów,
może spowodować wzrost efektywności przetwarzania. Wzrost ten może wynikać z
dodatkowego wielokrotnego wykorzystania tych samych wierszy i kolumn macierzy w
stopniu zależnym od liczby wyników liczonych przez wątek. Przykładowo, wątek obliczając 2
wyniki cząstkowe potrzebuje tylko 50% danych więcej, gdyż może mnożyć ten sam fragment
wiersza macierzy A przez 2 różne fragmenty kolumn macierzy B.

Poniższe rozwiązanie proponuje zwiększenie pracy wykonywanej przez wątek, co przekłada
się na większą intensywność obliczeń, co oznacza że każdy wątek oblicza 4 elementy
macierzy wynikowej zamiast 1.

```c++
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA_Modified(float* C, float* A, float* B, int width) {
// Indeksy wątków i bloków
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
// Pamięć współdzielona na kafelki macierzy A i B
__shared__ float shared_A[2][BLOCK_SIZE][BLOCK_SIZE];
__shared__ float shared_B[2][BLOCK_SIZE][BLOCK_SIZE];
// Inicjalizacja akumulatorów wyników
float Csub[4] = {0};
// Wyznaczenie początkowych pozycji kafelków dla bloku
int row_start_A[2] ={
by * BLOCK_SIZE * width,
by * BLOCK_SIZE * width + (width * width / 2)
};
int col_start_B[2] = {
bx * BLOCK_SIZE,
bx * BLOCK_SIZE + (width / 2)
};
// Iteracja po kafelkach macierzy A i B
for (int k = 0; k < width; k += BLOCK_SIZE) {
// Załaduj kafelki do pamięci współdzielonej dla obu zestawów macierzy
// kafelek z pierwszej części macierzy A
shared_A[0][ty][tx] = A[row_start_A[0] + k + ty * width + tx];
// kafelek z pierwszej części macierzy B
shared_B[0][ty][tx] = B[k * width + col_start_B[0] + ty * width + tx];
// kafelek z drugiej części macierzy A
shared_A[1][ty][tx] = A[row_start_A[1] + k + ty * width + tx];
// kafelek z drugiej części macierzy B
shared_B[1][ty][tx] = B[k * width + col_start_B[1] + ty * width + tx];
// Synchronizacja wątków, aby upewnić się, że kafelki są w pełni załadowane
__syncthreads();
// Obliczenia na aktualnych kafelkach
#pragma unroll
for (int i = 0; i < BLOCK_SIZE; ++i) {
// Pobierz wartości z kafelków A i B
float val_A[2] = {shared_A[0][ty][i], shared_A[1][ty][i]};
float val_B[2] = {shared_B[0][i][tx], shared_B[1][i][tx]};
// Akumulacja wyników dla czterech kwadrantów
Csub[0] += val_A[0] * val_B[0]; // wynik dla pierwszego kwadrantu
Csub[1] += val_A[0] * val_B[1]; // wynik dla drugiego kwadrantu
Csub[2] += val_A[1] * val_B[0]; // wynik dla trzeciego kwadrantu
Csub[3] += val_A[1] * val_B[1]; // wynik dla czwartego kwadrantu
}
// Synchronizacja wątków przed załadowaniem kolejnych kafelków
__syncthreads();
}
// Zapis wyników do pamięci globalnej
// początkowy indeks dla bieżącego bloku w macierzy wynikowej
//pierwszy kwadrant
int c = by*BLOCK_SIZE * width + bx * BLOCK_SIZE; C[c + ty * width + tx] = Csub[0];
C[c + (width/2) + ty * width + tx] = Csub[1]; // drugi kwadrant
C[c + (width/2) * width + ty * width + tx] = Csub[2]; //trzeci kwadrant
C[c + (width/2) * width + (width/2) + ty * width + tx] = Csub[3]; //czwarty kwadrant
}
```
Zmianie ulegają rozmiar siatki, z jaką uruchamiany jest program:
```c++
dim3 grid_Modified(ceil((float)dimsB.x/(threads.x*2)), ceil((float)dimsA.y/(threads.y*2)));
```
nasza metoda polega na podziale obszaru roboczego na 4 kwadranty, musimy zmniejszyć
wymiary siatki o połowę w porównaniu z oryginalnym kodem.

Rysunek nr. 1.: Pokazuje które fragmenty macierzy A i B będą pobierane przez przykładowy
blok do pamięci współdzielonej, oraz które fragmenty macierzy C będą przez ten blok obliczane.

![image](https://github.com/user-attachments/assets/97d893bc-dad0-4be3-97d0-c8ce98c2cfbb)

Z macierzy A i B ładujemy do pamięci współdzielonej podmacierze - jednorazowo pobierana
jest większa ilość danych, co zmniejsza czas i potrzebę dostępu do pamięci globalnej i
zwiększa efektywność obliczeń.

Rysunek nr. 2. przedstawia kombinacje użytych fragmentów macierzy A i B do obliczenia
wartości cząstkowych danych fragmentów macierzy C

![image](https://github.com/user-attachments/assets/f4bbc8be-224b-4b78-876f-99bfcd8c8c35)

Jednocześnie do pamięci ładujemy po 1 z elemencie z każdej z 2 podmacierzy A i po 1
elemencie z każdej z 2 podmacierzy B. Równolegle realizujemy na tych danych 4 obliczenia
i w rezultacie otrzymujemy 4 elementy macierzy wynikowej C. Nadal korzystamy z podejścia
kafelkowego, lecz z lepszą efektywnością wykorzystania danych.

# Testy i pomiary

W celu testów macierze zostaną wypełnione za pomocą funkcji LinearInt:

```c++
void LinearInit(float* data, int size, bool reverse){
 if (reverse) {
 for (int i = 0; i < size; ++i) {
 data[i] = static_cast<float>(size - i);
 }
 }
 else {
 for (int i = 0; i < size; ++i) {
 data[i] = static_cast<float>(i);
 }
 }
}
```
Funkcja LinearInit wypełnia tablicę liczbami rosnącymi lub malejącymi, w zależności od
wartości parametru reverse. Dla reverse = false, wartości są wypełniane rosnąco od
0 do size-1. Dla reverse = true, wartości są wypełniane malejąco od size-1 do 0.
Dla przykładu, dla macierzy 3x3 dostaniemy:
Dla macierzy A:         Dla macierzy B:
0, 1, 2,                    8, 7, 6
3, 4, 5,                    5, 4, 3
6, 7, 8                     2, 1, 0

Do testów wydajności zostaną użyte macierze o rozmiarach:
1024x1024, 1600x1600, 2048x2048, 3200x3200

Mniejsze rozmiary (1024x1024) pozwalają na testowanie podstawowej wydajności i
odpowiedzi systemu na niewielkie obciążenie, natomiast większe macierze (3200x3200)
umożliwiają ocenę efektywności algorytmu przy większych danych, uwzględniając wpływ
pamięci, przepustowości i obliczeń równoległych. Testowanie różnych rozmiarów pomaga
zrozumieć skalowalność algorytmu na różnych konfiguracjach sprzętowych.

Wyniki pomiarów są to średnie z 10 wyników

```c++
int nIter = 10;
 for (int j = 0; j < nIter; j++) {
 if (block_size == 16) {
 MatrixMulCUDA_Modified<16>
 << <grid_Modified, threads, 0, stream >> > (d_C, d_A, d_B,
dimsA.x);
 }
 else {
 MatrixMulCUDA_Modified<32>
 << <grid_Modified, threads, 0, stream >> > (d_C, d_A, d_B,
dimsA.x);
 }
 }
```
Pomiar czasu nie jest robiony dla każdej iteracji z osobna ale dla całego procesu
wykonywania 10 iteracji naraz, więc obliczamy średnią poprzez podzielenie całkowitego
czasu przez liczbę iteracji
```c++
float msecPerMatrixMul = msecTotal / nIter;
```
Na podstawie średniego czasu jednej iteracji oraz liczby operacji matematycznych, które
zostały przeprowadzone (w tym przypadku operacji mnożenia macierzy), obliczana jest
wydajność w gigaFlopach:
```c++
double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
static_cast<double>(dimsA.y) * static_cast<double>(dimsB.x);
double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
```
## Kod obliczający 1 wynik (Blok 16x16)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) |
|------------------|--------------------|-------------|
| 1024x1024       | 1000,22            | 2,147       |
| 1600x1600       | 881,03             | 9,298       |
| 2048x2048       | 1111,42            | 15,458      |
| 3200x3200       | 808,75             | 81,034      |


![image](https://github.com/user-attachments/assets/bbe63c13-75cc-468b-b7ba-0f9c3e844cd6)


## Kod obliczający 1 wynik (Blok 32x32)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) |
|------------------|--------------------|-------------|
| 1024x1024       | 1084,36            | 1,980       |
| 1600x1600       | 1268,53             | 6,458       |
| 2048x2048       | 1292,25             | 13,398      |
| 3200x3200       | 1272,74             | 51,492      |


![image](https://github.com/user-attachments/assets/5f39d7da-34d8-4f87-bc4c-bc4f77b67466)


## Kod obliczający 4 wyniki (Blok 16x16)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) | Przyśpieszenie względem NVIDIA |
|------------------|--------------------|-------------|---------------------------------|
| 1024x1024       | 2131,69            | 1,007       | 2,13x |
| 1600x1600       | 2201,01             | 3,722       | 2,5x |
| 2048x2048       | 2400,38             | 7,157      | 2,16x |
| 3200x3200       | 1824,79             | 35,914      | 2,26x |


![image](https://github.com/user-attachments/assets/e3f76068-4627-46f0-8a88-da786da864d1)


## Kod obliczający 4 wyniki (Blok 32x32)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) | Przyśpieszenie względem NVIDIA |
|------------------|--------------------|-------------|---------------------------------|
| 1024x1024       | 2183,17            | 0,982       | 2,01x |
| 1600x1600       | 2315,42             | 3,538       | 1,83x |
| 2048x2048       | 2720,75             | 6,314      | 2,11x |
| 3200x3200       | 2725,62             | 24,044      | 2,14x |


![image](https://github.com/user-attachments/assets/71d885b1-60d0-4aa4-93a5-c692b96d3a84)


# Wnioski

Obliczanie 4 wyników na wątek osiąga większą wydajność w każdej sytuacji względem
obliczania 1 wyniku na wątek.
Wynika to z lepszego wykorzystania pamięci współdzielonej oraz większej liczby operacji
przypadających na wątek, co prowadzi do lepszego wykorzystania równoległości.
Dla dużych macierzy ilość danych przenoszonych z pamięci globalnej do współdzielonej
wzrasta, co zwiększa koszty transferu pamięci. Jednocześnie współczynnik ponownego
użycia danych z kafelków maleje, co prowadzi do większej liczby operacji odczytu i zapisu w
pamięci globalnej.

### Wpływ rozmiaru bloku:

  ● Blok 32x32 osiąga wyższą wydajność w każdej konfiguracji. Wynika to z większej
  liczby wątków na blok, co lepiej wykorzystuje zasoby GPU. Wykazuje znacznie
  lepszą efektywność współdzielonej pamięci (70% vs. 42.31%). W większych blokach
  więcej danych może być przetwarzanych lokalnie, co minimalizuje potrzebę dostępu
  do pamięci globalnej.
  
  ● Blok 16x16 ma niższą wydajność, szczególnie dla dużych macierzy, co może być
  spowodowane większym narzutem na synchronizację, mniejsza liczba wątków
  prowadzi do niewykorzystania pełnej przepustowości pamięci współdzielonej.
  Zatrzymania spowodowane zależnościami pamięci są znacznie wyższe w przypadku
  bloku 16x16 (np. 55% vs. 8.81% przy 4 wynikach). Wynika to z częstszych
  odczytów/zapisów do pamięci globalnej w mniejszych blokach.

# Potencjalny dalszy rozwój kodu

Z analizy wyników wynika, że warto zoptymalizować wykorzystanie pamięci współdzielonej
poprzez lepsze wyrównanie i minimalizację konfliktów w dostępie do danych, co pozwoli na
zwiększenie efektywności operacji. Dostosowanie liczby wątków i rejestrów do dostępnych
zasobów GPU może poprawić obłożenie jednostek obliczeniowych i zmniejszyć
fragmentację. Istotnym usprawnieniem byłoby także wprowadzenie wektoryzacji dostępu do
pamięci globalnej, co skróciłoby czas transferu danych pomiędzy różnymi poziomami
pamięci. Rozpakowanie pętli w sekcjach obliczeniowych pozwoli na pełniejsze
wykorzystanie równoległości obliczeń, redukując narzuty związane z kontrolą przepływu.
Warto także rozważyć dynamiczne dostosowanie rozmiarów bloków i kafelków w zależności
od rozmiaru macierzy, aby uzyskać optymalny balans pomiędzy kosztami transferu danych a intensywnością obliczeń.

# Dodatkowe eksperymenty i testy

Dodatkowo sprawdziliśmy czy zwiększenie liczby wyników do 8 i 16 na wątek poprawi wydajność obliczeń.

## Kod obliczający 8 wyniki (Blok 16x16)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) | Przyśpieszenie względem NVIDIA |
|------------------|--------------------|-------------|---------------------------------|
| 1024x1024       | 2611,32            | 0,822       | 2,61x |
| 1600x1600       | 2732,09             | 2,998       | 3,1x |
| 2048x2048       | 3160,1             | 5,436      | 2,84x |
| 3200x3200       | 2318,64             | 28,265      | 2,87x |

![image](https://github.com/user-attachments/assets/68d4e956-fbdb-44d3-9571-a89a047d9dcc)

## Kod obliczający 8 wyniki (Blok 32x32)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) | Przyśpieszenie względem NVIDIA |
|------------------|--------------------|-------------|---------------------------------|
| 1024x1024       | 2236,54            | 0,960       | 2,06x |
| 1600x1600       | 2214,24             | 3,700       | 1,75x |
| 2048x2048       | 2710,94            | 6,337     | 2,1x |
| 3200x3200       | 2723,64             | 24,062      | 2,14x |

![image](https://github.com/user-attachments/assets/4b885591-ecd6-4da8-b1a5-332c7faa4cea)

Jak liczymy 8 wyników to oczywiście potrzebujemy więcej danych.

W przypadku macierzy A operujemy na 2 kafelkach w poziomie, ponieważ obliczenia są
oparte na transpozycji macierzy. Każdy wątek przetwarza elementy z dwóch różnych wierszy
macierzy A dla różnych wyników w macierzy wynikowej, co oznacza, że liczba fragmentów z
A nie zmienia się, niezależnie od liczby wyników.

Natomiast w przypadku macierzy B, ponieważ obliczamy więcej wyników na wątek (8),
musimy wziąć pod uwagę różne fragmenty kolumn B dla różnych wyników macierzy. W
przypadku 8 wyników, każdy wątek przetwarza dane z 4 różnych fragmentów kolumn, co
odpowiada 4 różnym częściom macierzy B. Dlatego liczba fragmentów B wzrasta, aby
pokryć wszystkie obliczenia, a liczba fragmentów A pozostaje stała, ponieważ dla A nie
zmienia się liczba wierszy do przetwarzania.

## Kod dla 8 wyników na wątek:

```c++
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA_Modified(float* C, float* A, float* B, int width) {
 int bx = blockIdx.x;
 int by = blockIdx.y;
 int tx = threadIdx.x;
 int ty = threadIdx.y;
 __shared__ float shared_A[2][BLOCK_SIZE][BLOCK_SIZE];
 __shared__ float shared_B[4][BLOCK_SIZE][BLOCK_SIZE];
 float Csub[8] = {0};
 int row_start_A[2] = {
 by * BLOCK_SIZE * width,
 by*BLOCK_SIZE * width + (width * width / 2)
 };
 int col_start_B[4] = {
 bx * BLOCK_SIZE,
 bx * BLOCK_SIZE + (width / 4),
 bx * BLOCK_SIZE + (width / 2),
 bx * BLOCK_SIZE + (3 * width / 4)
 };
 for (int k = 0; k < width; k += BLOCK_SIZE) {
 shared_A[0][ty][tx] = A[row_start_A[0] + k + ty * width + tx];
 shared_A[1][ty][tx] = A[row_start_A[1] + k + ty * width + tx];
 #pragma unroll
 for (int i = 0; i < 4; ++i) {
 shared_B[i][ty][tx] = B[k * width + col_start_B[i] + ty * width + tx];
 }
 __syncthreads();
 #pragma unroll
 for (int i = 0; i < BLOCK_SIZE; ++i) {
 float val_A[2] = {shared_A[0][ty][i], shared_A[1][ty][i]};
 float val_B[4] = {
 shared_B[0][i][tx],
 shared_B[1][i][tx],
 shared_B[2][i][tx],
 shared_B[3][i][tx]
 };

 Csub[0] += val_A[0] * val_B[0];
 Csub[1] += val_A[0] * val_B[1];
 Csub[2] += val_A[1] * val_B[0];
 Csub[3] += val_A[1] * val_B[1];
 Csub[4] += val_A[0] * val_B[2];
 Csub[5] += val_A[0] * val_B[3];
 Csub[6] += val_A[1] * val_B[2];
 Csub[7] += val_A[1] * val_B[3];
 }
 __syncthreads();
 }
 int c = by * BLOCK_SIZE * width + bx * BLOCK_SIZE;
 C[c + ty * width + tx] = Csub[0];
 C[c + (width / 4) + ty * width + tx] = Csub[1];
 C[c + (width / 2) * width + ty * width + tx] = Csub[2];
 C[c + (width / 2) * width + (width / 4) + ty * width + tx] = Csub[3];
 C[c + (width / 2) + ty * width + tx] = Csub[4];
 C[c + (3 * width / 4) + ty * width + tx] = Csub[5];
 C[c + (width / 2) * width + (width / 2) + ty * width + tx] = Csub[6];
 C[c + (width / 2) * width + (3 * width / 4) + ty * width + tx] = Csub[7];
}
```
Tak jak w przypadku dla 4 wyników na wątek, musimy zmniejszyć rozmiar gridu
```c++
dim3 grid_Modified(ceil((float)dimsB.x / (threads.x * 4)), ceil((float)dimsA.y / (threads.y * 2)));
```
W przypadku obliczania 8 wyników na wątek, każdy wątek przetwarza większą liczbę
elementów z macierzy A (2 fragmenty z A) oraz macierzy B (4 fragmenty z B).

Aby odpowiednio dopasować liczbę bloków do nowych obliczeń:

  ● Zwiększamy liczbę bloków w osi x (dla macierzy B) o czynnik 4, Aby zapewnić pełne
  pokrycie wszystkich elementów macierzy B, trzeba zmniejszyć liczbę wątków w
  kierunku X, co wymaga zwiększenia rozmiaru siatki (grid) w tym kierunku.
  
  ● Liczba bloków w osi y (dla macierzy A) pozostaje bez zmian, ponieważ każdy wątek
  nadal przetwarza 2 fragmenty z A w pionie.

Stąd zmiana w siatce polega na dostosowaniu liczby bloków w poziomie do większej liczby
fragmentów z macierzy B, które muszą być przetworzone przez każdy wątek.

## Kod obliczający 16 wyniki (Blok 16x16)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) | Przyśpieszenie względem NVIDIA |
|------------------|--------------------|-------------|---------------------------------|
| 1024x1024       | 3709,15            | 0,579       | 3,71x |
| 1600x1600       | 4276,93             | 1,915       | 4,85x |
| 2048x2048       | 4326,62             | 3,971       | 3,89x |
| 3200x3200       | 4816,17              | 13,607     | 5,96x |

![image](https://github.com/user-attachments/assets/31f232c8-1d42-4099-9bb1-212814b47f50)

## Kod obliczający 16 wyniki (Blok 32x32)

| Rozmiar macierzy | Wydajność (GFlop/s) | Czas (msec) | Przyśpieszenie względem NVIDIA |
|------------------|--------------------|-------------|---------------------------------|
| 1024x1024       | 3258,12            | 0,659       | 3x |
| 1600x1600       | 3544,11             | 2,311       | 2,79x |
| 2048x2048       | 4055,11             | 4,237       | 3,14x |
| 3200x3200       | 4571,82              | 14,335     | 3,59x |

![image](https://github.com/user-attachments/assets/ae2f3a5e-5aa3-4c1e-b496-1315eaecc77e)

### Kod programu dla 16 wyników na wątek:

```c++
template <int BLOCK_SIZE>
__global__ void MatrixMulCUDA_Modified(float* C, float* A, float* B, int width) {
int bx = blockIdx.x;
int by = blockIdx.y;
int tx = threadIdx.x;
int ty = threadIdx.y;
__shared__ float shared_A[4][BLOCK_SIZE][BLOCK_SIZE];
__shared__ float shared_B[4][BLOCK_SIZE][BLOCK_SIZE];
float Csub[16] = {0};
int row_start_A[4] = {
by * BLOCK_SIZE * width,
by * BLOCK_SIZE * width + (width * width / 4),
by * BLOCK_SIZE * width + (width * width / 4),
by * BLOCK_SIZE * width + (width * width / 4)
};
int col_start_B[4] = {
bx * BLOCK_SIZE,
bx * BLOCK_SIZE + (width / 4),
bx * BLOCK_SIZE + (width / 4),
bx * BLOCK_SIZE + (width / 4)
};
for (int k = 0; k < width; k += BLOCK_SIZE) {
#pragma unroll
for (int i = 0; i < 4; ++i) {
shared_A[i][ty][tx] = A[row_start_A[i] + k + ty * width + tx];
shared_B[i][ty][tx] = B[k * width + col_start_B[i] + ty * width + tx];
}
__syncthreads();
#pragma unroll
for (int i = 0; i < BLOCK_SIZE; ++i) {
float val_A[4] = {
shared_A[0][ty][i],
shared_A[1][ty][i],
shared_A[2][ty][i],
shared_A[3][ty][i]
};
float val_B[4] = {
shared_B[0][i][tx],
shared_B[1][i][tx],
shared_B[2][i][tx],
shared_B[3][i][tx]
};
Csub[0] += val_A[0] * val_B[0];
Csub[1] += val_A[0] * val_B[1];
Csub[2] += val_A[0] * val_B[2];
Csub[3] += val_A[0] * val_B[3];
Csub[4] += val_A[1] * val_B[0];
Csub[5] += val_A[1] * val_B[1];
Csub[6] += val_A[1] * val_B[2];
Csub[7] += val_A[1] * val_B[3];
Csub[8] += val_A[2] * val_B[0];
Csub[9] += val_A[2] * val_B[1];
Csub[10] += val_A[2] * val_B[2];
Csub[11] += val_A[2] * val_B[3];
Csub[12] += val_A[3] * val_B[0];
Csub[13] += val_A[3] * val_B[1];
Csub[14] += val_A[3] * val_B[2];
Csub[15] += val_A[3] * val_B[3];
}
__syncthreads();
}
int c = width * BLOCK_SIZE * by + BLOCK_SIZE * bx;
C[c + width * ty + tx] = Csub[0];
C[c + (width / 4) + width * ty + tx] = Csub[1];
C[c + (width / 4) * width + width * ty + tx] = Csub[2];
C[c + (width / 4) * width + (width / 4) + width * ty + tx] = Csub[3];
C[c + ((width / 4) * 2) + width * ty + tx] = Csub[4];
C[c + ((width / 4) * 3) + width * ty + tx] = Csub[5];
C[c + (width / 4) * width + ((width / 4) * 2) + width * ty + tx] = Csub[6];
C[c + (width / 4) * width + ((width / 4) * 3) + width * ty + tx] = Csub[7];
C[c + ((width / 4) * width) * 2 + width * ty + tx] = Csub[8];
C[c + ((width / 4) * width) * 2 + (width / 4) + width * ty + tx] = Csub[9];
C[c + ((width / 4) * width) * 3 + width * ty + tx] = Csub[10];
C[c + ((width / 4) * width) * 3 + (width / 4) + width * ty + tx] = Csub[11];
C[c + ((width / 4) * width) * 2 + ((width / 4) * 2) + width * ty + tx] = Csub[12];
C[c + ((width / 4) * width) * 2 + ((width / 4) * 3) + width * ty + tx] = Csub[13];
C[c + ((width / 4) * width) * 3 + ((width / 4) * 2) + width * ty + tx] = Csub[14];
C[c + ((width / 4) * width) * 3 + ((width / 4) * 3) + width * ty + tx] = Csub[15];
}
```
Tak jak w przypadku pozostałych programów, musimy zmniejszyć rozmiar siatki:
```c++
dim3 grid_Modified(ceil((float)dimsB.x/(threads.x * 4)), ceil((float)dimsA.y/(threads.y * 4)));
```
# Podsumowanie wersji kodów 8 i 16 wyników na wątek

Wersje z 4, 8 i 16 wynikami na wątek poprawiają wydajność, zwiększając wykorzystanie
jednostek obliczeniowych i pamięci współdzielonej, co skraca czas obliczeń. Większa liczba
wyników na wątek pozwala na lepsze wykorzystanie przepustowości pamięci i zmniejsza
czas oczekiwania na dane. Jednak takie podejście może prowadzić do większego zużycia
rejestrów i pamięci współdzielonej, co może ograniczyć liczbę wątków w jednym bloku, a
tym samym obniżyć ogólne obciążenie GPU. Ponadto, większa liczba wyników na wątek
może zwiększyć złożoność zarządzania pamięcią i synchronizacji wątków, co może w
niektórych przypadkach negatywnie wpłynąć na wydajność.

## Wpływ rozmiaru bloku

Blok 16x16 daje lepsze wyniki przy obliczaniu 8 i 16 wyników na wątek, ponieważ przy
większej liczbie wyników na wątek, rozmiar bloku 16x16 lepiej dopasowuje się do
dostępnych zasobów GPU. Mniejszy rozmiar bloku skutkuje mniejszym narzutem na
synchronizację w porównaniu do większych bloków, co jest korzystne przy przetwarzaniu
większej liczby wyników na wątek. W przypadku większych rozmiary bloków (32x32)
prowadzą do niższego wykorzystania pamięci współdzielonej, co z kolei skutkuje większymi
opóźnieniami przy dostępie do pamięci globalnej, zwłaszcza przy przetwarzaniu większej
liczby wyników na wątek.
Dodatkowo, przy blokach 16x16 liczba wątków na blok jest wyższa, co umożliwia lepsze
wykorzystanie równoległości w przypadku większej liczby wyników na wątek. Mniejszy blok
zmniejsza także obciążenie pamięci globalnej, umożliwiając efektywniejsze wykorzystanie
pamięci współdzielonej
