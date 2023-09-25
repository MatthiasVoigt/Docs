
## Person Correlation, Cosine similarity, Covariance, Dot product and Euclidean distance of Standardized Vectors

This page compares different similarity measures that can be used to compute the similarity of two vectors. We assume that the vectors are standardized meaning, they have zero mean and the standard deviaiton is one. It turns out with standardized vectors the measures are just variations of the dot product. There is no fundamental difference. 




#### Dot Product for vectors $X$ and $Y$ of length $N$. For the dot product we are just multiplying the components of each vector with each other.

$$\begin{align}
X \cdot Y = \text{Dot}(X,Y) = \sum_{i=0}^N{x_i y_i} 
\end{align}$$

#### Covariance for vectors $X$ and $Y$ of length $N$. Before multiplying the components, we subtract the mean and at the end we devide by the number of components. 

$$\begin{align}
\text{cov}(X,Y) = \frac{1}{N} \sum{(x_i-\mu_x) (y_i-\mu_y)}
\end{align}$$

with $\mu_x=\mu_y=0$:

$$\begin{align}
\text{cov}(X,Y) = \frac{1}{N} \sum{x_i y_i} = \frac{1}{N} ( X \cdot Y)
\end{align}$$

If the mean is zero, then the covariance is the same as the dot product divided by the number of compoents N. 


#### Pearson Correlation for vectors $X$ and $Y$ of length $N$:

$$\begin{align}
r = \text{Pearson}(X, Y) = \frac{\sum_{i=1}^{N} (x_i - \bar{x}) (y_i - \bar{y})}{\sqrt{\sum_{i=1}^{N} (x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{N} (y_i - \bar{y})^2}} 
\end{align}$$

using the standard deviation $\sigma=\sqrt{\frac{\sum(x_i-\mu)^2}{N}}$ or   $\sqrt{N} \sigma =\sqrt{{\sum(x_i-\mu)^2}}$

$$
r = \frac{\sum(x_i - \bar{x})(y_i-\bar{y})}{N \sigma_x \sigma_y}
$$

with $\mu_x=\mu_y=0$ and $\sigma_x = \sigma_y =1$, :

$$\begin{align}
r = \frac{1}{N} \sum{x_i y_i}=  \frac{1}{N} ( X \cdot Y)
\end{align}$$

For standardized vectors the person correlation and covariance are the same. 

#### Cosine Similarity for vectors $X$ and $Y$ of length $N$:

$$\begin{align}
\text{cos}(\theta)= \text{Cosine}(X, Y) = \frac{\sum_{i=1}^{N} x_i y_i}{\sqrt{\sum_{i=1}^{N} x_i^2} \sqrt{\sum_{i=1}^{N} y_i^2}}
\end{align}$$

with $\mu_x=\mu_y=0$ and $\sigma_x = \sigma_y =1$, :

$$\begin{align}
\text{Cosine}(X, Y) = \sum x_i y_i = X \cdot Y 
\end{align}$$

For standardized vectors the Cosine similarity and the Dot product are the same. 


#### Euclidean distance, L2-Norm

$$\begin{align}
\text{d}(X,Y) =  \sqrt{\sum(x_i-y_i)^2} 
\end{align}$$

$$ \begin{align*}
\text{d}^2(X,Y)  &=& \sum(x_i-y_i)^2  \\
    &=& \sum ( x_i^2 - 2x_i y_i + y_i^2) \\
\end{align*}$$

with $\sigma^2=\frac{1}{N}\sum(x_i-\mu)^2$ , $\mu=0$, $\sigma=1$ 

$\sum{x_i^2} = \sigma^2 N = N$

$$ \begin{align*}
    & = & 2N - 2\sum{x_i y_i} \\
    & = & 2N - 2 (X \cdot Y) 
\end{align*}$$

$$ \begin{align}
    \text{d}(X,Y) = \sqrt{ 2N - 2 (X \cdot Y)} \\
\end{align}$$


and the relation between pearson correlation $r$ and euclidean distance $d$: 

$$ \begin{align}
\text{d}^2(X,Y) &=& 2N - 2 (X \cdot Y)  2\nonumber \\
N - \frac{1}{2}\text{d}^2(X,Y) &=&  (X \cdot Y) \nonumber  \\
1 - \frac{1}{2N}\text{d}^2(X,Y) &=& \frac{1}{N} (X \cdot Y) \nonumber 
\end{align}$$

$$ \begin{align}
 r = 1 - \frac{\text{d}^2(X,Y)}{2N} 
\end{align}$$

### Summary:  

With standardized vectores, we can express the pearson correlation $r$ as a function of the dot product, the covariance, the cosine, and the euclidean distance:

Pearson correlation and covariance are then identical:
$$\begin{align}
r = \text{cov}(X,Y)
\end{align}$$

The dot product and the cosine distance are equal to the pearson correlation times $N$. 
$$\begin{align}
r = \frac{1}{N} (X \cdot Y) \\
r = \frac{1}{N} \text{cos}(X, Y)  \\
\end{align}$$

The relation between pearson correlation and euclidean distance is. 
$$\begin{align}
r = 1 - \frac{\text{d}^2(X,Y)}{2N} 
\end{align}$$



Here is example code to validate the relation between pearson correlation and euclidean distance numerically:
```python
import matplotlib.pyplot as plt

# Create three random vectors of length N=1000
x = np.random.rand(1000)
y = np.random.rand(1000)
z = np.random.rand(1000)

# Compute a = x+y and b = x+z
a = x + y
b = x + z

# Standardize a and b vectors
standardized_a = (a - np.mean(a)) / np.std(a)
standardized_b = (b - np.mean(b)) / np.std(b)

# Compute Pearson correlation
corr_coefficient, _ = pearsonr(standardized_a, standardized_b)

# Compute Euclidean distance D
D = euclidean(standardized_a, standardized_b)

# Compute r value
r = 1 - (D * D) / 2000


corr_coefficient, D, r

```


```
RESULT
(0.5164490784852466, 31.098261093339396, 0.5164490784852467)
```
The first and last number of the output matches as expected. 

