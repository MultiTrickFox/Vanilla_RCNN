

(Suggested Training - For ~50k samples)



<datasize> <batch_size> <epochs>	<other session params>

30000 300 10				drop=0.3
30000 250 10				drop=0.2
30000 200 10				drop=0.2
30000 200 20					(drop=0.1 by default)


45000 150 20				drop=0.3

45000 100 30				drop=0.2

next:
45000 100 50's				startadv=True (optional)
				



optional next: 25000 50 20 's
*(total epochs around 150 ~ 300)

