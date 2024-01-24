# Regolarizzazione

La Regolarizzazione server per evitare problemi di overfitting, standardizza i pesi più grandi riducendo la complessità del modello

Aggiunge una penalità ai pesi più grandi mettendo un nuovo termine nella funzione di costo del modello durante l'addestramento

L'iperparametro Lambda è inserito nella funzione di costo per permettere di decidere la forza con cui la regolarizzazione agirà. Se è zero l'azione è nulla.
L'ideale è cercare un valore di lambda in un range di potenze di 10 (idealmente tra 10^-4 e 10)

- REGOLARIZZAZIONE L2 (weight decay): il termine aggiunto è la somma dei quadrati dei pesi, costringendo il modello a minimizzare la funzione di costo per valore dei pesi più piccoli.
- REGOLARIZZAZIONE L1: il termine aggiunto è la somma del valore assoluto dei pesi, quindi i pesi più piccoli vengono ridotti a zero

Modelli regolarizzati:

- L2: Ridge
- L1: Lasso
- Entrambe: ElasticNet

L'ideale è usare entrambe le regolarizzazoni

Per regolarizzare è importante che i dati siano sulla stessa scala quindi usare Standardizzazione o Normalizzazione
