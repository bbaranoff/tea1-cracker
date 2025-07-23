# tea1-cracker

example :  

Generate a keystream from https://github.com/MidnightBlueLabs/TETRA_crypto  
then try to reverse it with tea1_opencl_cracker.py  

```bash
(py310) nirvana@legion:~/TETRA_crypto$ ./gen_ks 1 110 30 06 1 0 11111111
TEA1_reduced hn 110 mn 30 fn 6 tn 1 eck 11111111
93794818CBE58966A07735527239B647AB8B67F1DA02580355F40C0F5BE7C99331989E1030E3FE5D4174D98B881E7039282161FAC805
(py310) nirvana@legion:~/TETRA_crypto$ python prep.py 1 110 30 06 1 0 93794818CBE58966A07735527239B647AB8B67F1DA02580355F40C0F5BE7C99331989E1030E3FE5D4174D98B881E7039282161FAC805
Génération du keystream pour frame: tn=1, hn=110, mn=30, fn=6, sn=1, dir=0, ks=93794818
Potential Key found: 11111111 !
^CInterruption par l'utilisateur, fermeture du pool.
```
