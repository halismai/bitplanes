#include <iostream>
#include <cstddef>

int main()
{
  uint8_t c0 = 150;
  uint8_t c1 = 23;

  float cost[8];
  for(int i = 0; i < 8; ++i)
  {
    cost[i] =
        ((c1 & (1<<i)) >> i) -
        ((c0 & (1<<i)) >> i);
    printf("%g ", cost[i]);
  }
  printf("\n");

  uint8_t d = c1 ^ c0;
  int s = (c1 > c0) ? 1 : -1;
  for(int i = 0; i < 8; ++i)
    printf("%d ", s * (d & (1<<i)) >> i);
  printf("\n");

  return 0;
}

