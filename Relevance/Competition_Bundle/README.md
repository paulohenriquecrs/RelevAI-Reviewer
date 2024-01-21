# Relevance Competition Bundle

***
This folder has the sample competition bundle. The steps below will show how you can compile the bundle and then upload to Codabench to create your compeition.

### 1. Compile competiiton Bundle

#### - go to the competition bundle directory
```
cd M1-Challenge-Class-2024/Relevance/Competition_Bundle
```

#### - compile the bundle
```
python3 utilities/compile_bundle.py
```

this will create a zip file named `Relevance_Bundle.zip` in `M1-Challenge-Class-2024/Relevance/Competition_Bundle/`

### 2. Upload to Codabench

#### - Codabench account

Create a codabench account here: https://www.codabench.org/

#### - Upload bundle

Upload the zipped bundle in `Benchmarks/Management` -> `Upload`

***

Now that you know how you can create a Codabench competition website from this bundle, follow the steps below to make your own version of this bundle

### Modify the Competition Bundle

#### 1. Update Pages

Update the competition pages by putting upto-date information in each page. Find the pages in `M1-Challenge-Class-2024/Relevance/Competition_Bundle/pages/`

For inspiration, please check this competition pages: https://www.codabench.org/competitions/1357/

#### 2. Update Ingestion and Scoring Program

Update the ingestion and scoring program if needed

#### 3. Sample submission

provide a zipped sampled submission in the competition pages

#### 4. Sample submission

provide a starting kit to the participants


*** 
### Docker (Optional)

Note: On Codabench a submission is run inside a docker container. We have provided a container image with pre-installed packages. If you want to install additional libraries/packages for your submissions, you may want to create your own docker image and then use it in your competition.

The base image you can use is: `ihsaanullah/llm:latest`.

Check this link for more details about creating a docker image: https://github.com/ihsaan-ullah/create_a_codalab_challenge?tab=readme-ov-file#docker