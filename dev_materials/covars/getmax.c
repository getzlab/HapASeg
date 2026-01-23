#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <byteswap.h>

int main() {
   FILE* x = fopen("wgEncodeOpenChromFaireGm12878BaseOverlapSignal.fwb", "r");
   uint16_t max = 0;
   uint16_t buf;
   while(fread(&buf, 2, 1, x)) {
      buf = __bswap_16(buf);
      if(buf > max) { max = buf; printf("%d\n", max); }
   }
   return 0;
}
