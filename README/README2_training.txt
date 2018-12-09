

(Suggested Training - For ~50k samples)



<datasize> <batch_size> <epochs>	<other session params>

30000 200 50				drop=0.3
30000 150 50				drop=0.2




45000 300 10
45000 200 10

25000 100 10 ; 				startadv=True ; drop=0.3
25000 100 10 ; 				startadv=True ; drop=0.2
25000 100 10 ; 				startadv=True   	(drop=0.1 by default)

45000 50  50				


optional next: 25000 50 20 's
*(total epochs around 150 ~ 300)

