# Themenextraktion
## Anwendung
Das Skript wird ohne zusätzliche Kommandozeilenparameter gestartet.

Zu Beginn fragt das Skript interaktiv die gewünschten Einstellungen ab (z. B. Word-Embedding- und Topic-Modelling-Methode).

Die Trainingsdaten müssen im Ordner data/ liegen und entweder comcast_consumeraffairs_complaints.csv (Rohdaten) oder preprocessed_corpus.pkl (vorverarbeiteter Korpus) heißen.

## Informationen
Für jede Kombination aus Word-Embedding- und Topic-Modelling-Methode wird ein separater Ergebnisordner angelegt.

Jeder Ordner enthält:

- das beste trainierte Modell

- eine CSV-Datei mit allen getesteten Parametern und den zugehörigen Coherence Scores
