

(Suggested Training - For ~50k samples)



			> run sessions with: onlyloss=1,2,3,4 for whole training


<datasize> <batch_size> <epochs>	<other session params>

45000 300 10
45000 200 10

25000 100 10 ; 				startadv=True ; drop=0.3
25000 100 10 ; 				startadv=True ; drop=0.2
25000 100 10 ; 				startadv=True   	(drop=0.1 by default)

45000 50  50				


optional next: 25000 50 20 's
*(total epochs around 150 ~ 300)



			> run sessions with: onlyloss=<item>,<item>.. ; (suggested way is one item per training)


repeat same, with higher dropouts. (i.e. 0.5 downto 0.1 with smaller epochs.)

