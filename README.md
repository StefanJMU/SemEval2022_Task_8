# SemEval2022_Task_8
Proposal for SemEval 2022 Task 8: Similarity assessment of multilingual news articles

## Database setup with Docker

Download and unzip the files
* frequency_dump.zip
* embeddings_dump.zip

from <a href='https://drive.google.com/drive/folders/1IlIASnn1qvijD28DdDMmvdkoV5xrfNrM?usp=sharing'>here</a>.

Run in the directory harboring these files :

```console
$ docker run --name <your container name> -e POSTGRES_PASSWORD=root -p 5432:5432 -d postgres
```

When the container is up and running, run on **Linux**
```console
$ cat frequency_dump.txt | docker exec -i <your container name> psql -U postgres
```
```console
$ cat embeddings_dump.txt | docker exec -i <your container name> psql -U postgres
```

and on **Windows** run

```console
$ type frequency_dump.txt | docker exec -i <your container name> psql -U postgres
```
```console
$ type embeddings_dump.txt | docker exec -i <your container name> psql -U postgres
```
