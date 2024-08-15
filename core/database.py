from cfg import CLASS_NAME, ROOT, COLLECTION_PATH
from cfg import os ,tqdm, np, Image, chromadb, json
from cfg import embedding_function

def get_files_path(path):
    files_path = []
    for label in CLASS_NAME:
        label_path = path + "/" + label
        filenames = os.listdir(label_path)
        for filename in filenames:
            filepath = label_path + '/' + filename
            files_path.append(filepath)
    return files_path

data_path = f'{ROOT}/train'
files_path = get_files_path(path=data_path)

def get_single_image_embedding(image : np.ndarray):
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)

def add_embedding(collection, files_path):
    ids = []
    embeddings = []
    for id_filepath, filepath in tqdm(enumerate(files_path)):
        ids.append(f'id_{id_filepath}')
        image = Image.open(filepath)
        # Convert Image to nparray
        image = np.array(image)
        embedding = get_single_image_embedding(image=image)
        embeddings.append(embedding.tolist())

    try:    
        collection.add(
            embeddings = embeddings,
            ids = ids
        )
    except Exception as e:
        print(e)

    __save_collection(collection, os.path.join(COLLECTION_PATH,"l2"))
    return collection

def __save_collection(collection, save_path):
    print('start saving ')
    """
        Save collection
        * save collection.metadata
        * save features embeddings
        * save ids
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    metadata_path = os.path.join(save_path, 'metadata.json')
    with open(metadata_path, 'w') as metadata_file:
        json.dump(collection.metadata, metadata_file)

    embeddings, ids = collection.get_all_embeddings_and_ids()
    embeddings_path = os.path.join(save_path, 'embeddings.npy')
    np.save(embeddings_path, embeddings)
    ids_path = os.path.join(save_path, 'ids.json')
    with open(ids_path, 'w') as ids_file:
        json.dump(ids, ids_file)

def load_collection(client, save_path, collection_name):
    """
        Collection loading: Return Collection
        * load collection
        * load features embeddings
        * idss
    """
    
    metadata_path = os.path.join(save_path, 'metadata.json')
    with open(metadata_path, 'r') as metadata_file:
        metadata = json.load(metadata_file)

    # Create the collection with the loaded metadata
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata=metadata
    )

    # Load the embeddings and ids
    embeddings_path = os.path.join(save_path, 'embeddings.npy')
    embeddings = np.load(embeddings_path)
    ids_path = os.path.join(save_path, 'ids.json')
    with open(ids_path, 'r') as ids_file:
        ids = json.load(ids_file)

    # Add the embeddings back to the collection
    collection.add(
        embeddings=embeddings, 
        ids=ids)

    return collection

def main():
    print('>>>>>>>>>>>>>>>>>>> Start Add Embedding ...')
    HNSW_SPACE = "hnsw:space"
    chroma_client = chromadb.Client()
    l2_collection = chroma_client.get_or_create_collection(
        name="l2_collection",
        metadata={"hnsw:space": "l2"} 
    )
    print('>>>>>>>>>>>>>>>>>>> Start Add l2_collection Embedding ...')
    collection = add_embedding(l2_collection, files_path)
    __save_collection(collection, os.path.join(COLLECTION_PATH,"l2"))

    cosine_collection = chroma_client.get_or_create_collection(
        name="cosine_collection",
        metadata={HNSW_SPACE: "cosine"})
    
    print('>>>>>>>>>>>>>>>>>>> Start Add cosine_collection Embedding ...')
    cosine_collection = add_embedding(cosine_collection, files_path)
    __save_collection(cosine_collection, os.path.join(COLLECTION_PATH,"cosine"))
    
    print('>>>>>>>>>>>>>>>>> Completed saving collection ...')

if __name__ == "__main__":
    main() 