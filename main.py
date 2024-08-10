
from utilies import ROOT, plot_results
from score import get_l1_score 
from score import get_l2_score
from score import get_cosine_similarity_score
from score import get_correlation_coefficient_score

if __name__ == "__main__":
    root_img_path = f"{ROOT}/train/"
    query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
    size = (640, 640)
    query, ls_path_score = get_correlation_coefficient_score(root_img_path, query_path, size)
    plot_results(query_path, ls_path_score, reverse=False)