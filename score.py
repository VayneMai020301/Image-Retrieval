
import os
from distance import  absolute_difference, mean_square_difference, cosine_similarity, correlation_coefficient
from utilies import read_image_from_path, folder_to_images, CLASS_NAME

def get_l1_score(root_img_path, query_path, size):
    """
    * Return 
    """
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) 


    rates = absolute_difference(query, images_np)
    ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score


def get_l2_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) 


    rates = mean_square_difference(query, images_np)
    ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score

def get_cosine_similarity_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) 


    rates = cosine_similarity(query, images_np)
    ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score



def get_correlation_coefficient_score(root_img_path, query_path, size):
    query = read_image_from_path(query_path, size)
    ls_path_score = []
    for folder in os.listdir(root_img_path):
        if folder in CLASS_NAME:
            path = root_img_path + folder
            images_np, images_path = folder_to_images(path, size) 


    rates = correlation_coefficient(query, images_np)
    ls_path_score.extend(list(zip(images_path, rates)))
    return query, ls_path_score