1. Preparing the required information of compounds.

2. Predicting the compounds' EI-MS spectra using NEIMS.

3. Generating an SQLite library：

   ![schema](https://github.com/Qiong-Yang/FastEI/blob/main/lib/schema.png)

    including identifiers(COMPID), SMILES, molecular mass weight (EXACTMOLWT), and predicted spectra(MZS and INTENSITYS).

4. Embedding the predicted spectra using the trained Word2vec model.

5. Adding the embeddings into an HNSW index.
