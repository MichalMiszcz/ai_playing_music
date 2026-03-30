# ai_playing_music

To repozytorium jeset przeznaczone do projektu generowania muzyki na podstawie zapisu nutowego.

W tym momencie pracuję nad modelem generującym plik .midi na podstawie pliku .png zawierającego zapis nutowy.

W celu wykorzystania konwertera należy pobrać program *MuseScore* oraz dodać go do zmiennych środowiskowych.

# Uruchomienie projektu

W celu korzystania z generowania nut należy pobrać program _MuseScore Studio 4_. Po pobraniu należy dodać ten program
do zmiennych środowiskowych.



## Opis folderów

- [src](src) - folder źródłowy 
  - [all data](src/all_data) - folder zawierający dane służące do uczenia modelu
  - [csv](src/csv) - folder zawierający dane opisujące wyliczone wyniki dokładności modelu
  - [music_program](src/music_program) - folder zawierający skrypty napisane w języku python, które zawierają konstrukcję
  modelu, dataloader oraz sekwencję uczenia
  - [new_data_generator](src/new_data_generator) - folder zawierający skrypty w python-ie służące do generowania nowych plików
  MIDI oraz nowych nut, które służą do uczenia modelu
  - [test](src/test) - folder zawierający pliki służące do testowania modelu lub nowych funkcji
  - [utils](src/utils) - folder zawierający pliki, używane w wielu programach na przestrzeni projektu