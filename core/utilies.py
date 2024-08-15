from cfg import Image, np , os, plt,mpimg
                                    
def read_image_from_path(path, size):
    im = Image.open(path).convert('RGB').resize(size)
    return np.array(im)

def folder_to_images(folder, size):
    """
        Return all of Image's information:
        * image's path: numpy array
        * image data: numpy array shape (num image, h, w, 3)
    """
    list_dir = [folder + '/'+ name for name in os.listdir(folder)]
    images_np = np.zeros(shape=(len(list_dir), *size, 3))
    images_path = []
    for i, path in enumerate(list_dir):
        images_np[i] = read_image_from_path(path, size)
        images_path.append(path)
        
    images_path = np.array(images_path)
    return images_np, images_path


def plot_results(query_path, results, reverse=False):
    """
    Plots the query image along with the top N results.
    
    Parameters:
    - query_path: Path to the query image.
    - results: A list of tuples, where each tuple contains the image path and the label.
    - reverse: If True, reverse the order of the results.
    """
    if reverse:
        results = results[::-1]
    
    n_results = len(results)
    
    # Define the number of rows and columns based on the number of images
    n_cols = 5                                          # Set the maximum number of columns
    n_rows = 1 + (n_results // n_cols)                  # +1 for the query image

    _, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # Plot the query image
    query_img = mpimg.imread(query_path)
    axes[0, 0].imshow(query_img)
    axes[0, 0].axis('off')
    axes[0, 0].set_title(f"Query Image: {query_path.split('/')[-1]}")

    # Plot the results
    for i, (img_path, label) in enumerate(results):
        row = (i + 1) // n_cols
        col = (i + 1) % n_cols
        img = mpimg.imread(img_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Top {i + 1}: {label}")
    
    for i in range(n_results + 1, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()