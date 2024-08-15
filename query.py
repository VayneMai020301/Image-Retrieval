
from core.cfg import np, os, Image, argparse, chromadb, plt
from core.cfg import embedding_function
from core.database import load_collection
from core.cfg import COLLECTION_PATH, root_img_path
from core.utilies import get_files_path,plot_chromadb_results

__doc__=[
    "query image in database with chromadb"
]
files_path = get_files_path(path=root_img_path)    
    
def __get_single_image_embedding(image : np.ndarray):
    """
        * Return feature of query image
    """
    embedding = embedding_function._encode_image(image=image)
    return np.array(embedding)

def __search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = __get_single_image_embedding(query_image)
    results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results 
    )
    return results

def main():
    parser = argparse.ArgumentParser(description="Choose options")
    parser.add_argument('-m', type = str, choices=[ 'l2', 'cosine'], required=True, default='l2',
                        help="Select score:  l2, cosine")
    parser.add_argument('-path', 
                        type = str,
                        choices=['data\test\African_crocodile\n01697457_18534.JPEG'], 
                        required=True,
                        default = 'data\test\African_crocodile\n01697457_18534.JPEG',
                        help="Path of image")
    parser.add_argument('-top', 
                        type = int,
                        choices=[5], 
                        required=True,
                        default=5,
                        help="Number of image with best scores")
    
    args = parser.parse_args()

    test_path = args.path
    client = chromadb.Client()
    if args.m == 'l2':
        l2_collection = load_collection(client,'l2_collection','l2', COLLECTION_PATH)
        results = __search(image_path=test_path, collection = l2_collection, n_results=args.top)

    elif args.m == 'cosine':
        cosine_collection = load_collection(client,'cosine_collection','cosine', COLLECTION_PATH)
        results = __search(image_path=test_path, collection = cosine_collection, n_results=args.top)

    
    plot_chromadb_results(image_path=test_path, files_path = files_path, results = results)
if __name__ == "__main__":
    main()
    