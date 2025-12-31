# neuronovae

$$
\cos x=\sum_{k=0}^{\infty}\frac{(-1)^k}{(2k)!}x^{2k}
$$

The final RGB image is rendered as:
$$
RGB(t,y,x) = I(t,y,x)+\sum_{r}^{N}\Delta_r(t,y,x)
$$
for each roi $r$ for colormap $N$ where $I$ is the initial grayscale image,
and $\Delta_r$ is the chromatic deviation for each roi, defined as:
$$ 
\Delta_r(t,y,x) = a_r(t,y,x) * w_i(y, x) * g(t, y, x) * (C_r(t)-1)
$$
