# Image-Retrieval
Query Image in Database
### How to run 
1. Clone this repository 
```
git clone 
```
2. Setup Environemnt
```
pip install 
pip install
```

### Retrieval with raw image features
* In this method, we approach calculate distance of each pair image and give the pair have best of of score
* Score is distance followed formula below

$$L1 (\vec{a}, \vec{b}) = \sum_{i=1}^N {|a-b|}$$
$$L2 (\vec{a}, \vec{b}) = \sqrt{\sum_{i=1}^N {(a-b)^2}}$$


$$\text{cosine similarity}(\vec{a}, \vec{b}) = \frac{a \cdot b}{||a|| ||b||} = \frac{\sum_{i=1}^N(a_ib_i)}{\sqrt{\sum_{i=1}^N a_i ^2} \sqrt{\sum_{i=1}^N b_i ^2}}$$

$$\text{r} = \frac{X[(X- \mu_X)(Y- \mu_Y)]}{\sigma_X \cdot \sigma_Y} =  \frac{\sum (x_i - \mu_X)(y_i - \mu_Y)}{\sqrt{\sum (x_i - \mu_X)^2 \sum (y_i - \mu_Y)^2}}$$


### Retrieval with extract features (using CLIP to extract features from image)
