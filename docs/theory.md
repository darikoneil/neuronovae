# Conceptual model

This visualization framework separates luminance from chromatic information. The 
underlying imaging signal (e.g. fluorescence intensity) is rendered purely as grayscale 
and is responsible for all perceived brightness. Regions of interest (ROIs) do not 
alter brightness; instead, they introduce color as a deviation from the grayscale axis. 
This design choice ensures that the displayed image faithfully preserves the original 
intensity structure while allowing ROIs to be highlighted without obscuring the 
background signal.

Each ROI is represented by a spatial weight map, which defines where the ROI 
contributes, and a time-varying activity scalar, which defines how strongly it 
contributes. These two components are explicitly decoupled: spatial weights determine 
where color can appear, while activity determines how much color is applied. Colors 
themselves are drawn from colormaps evaluated as a function of ROI activity, allowing 
intuitive mappings such as low activity → dark colors and high activity → saturated 
colors.

# Intensity-gated color modulation

Color is applied only when pixel intensity exceeds a defined baseline. This is 
implemented through an intensity-dependent gating function that smoothly ramps 
from zero to one over a specified intensity range. At or below baseline, pixels 
remain strictly grayscale; above baseline, chromatic contributions are gradually 
enabled. This prevents spurious coloring of background or noise and ensures that 
baseline frames or regions appear identical to the original data.

Crucially, intensity is not used as an opacity in the graphics sense. Instead, it 
modulates the magnitude of chromatic deviation from grayscale. As a result, black 
pixels remain black, white pixels remain white, and color does not introduce artificial 
contrast. Bright, active ROIs become more saturated, but never brighter than the 
underlying signal.

# Handling overlap and multiple ROIs

When multiple ROIs overlap, their chromatic contributions are summed additively in 
color space. Because each ROI contributes a deviation from the same grayscale 
reference, overlap results in intuitive color mixing without order dependence or 
competitive suppression. Neuropil regions remain grayscale regardless of ROI density, 
and ROI size does not bias perceived intensity or color strength.

This approach avoids the pitfalls of traditional alpha compositing, such as
cumulative darkening, arbitrary ordering effects, or implicit renormalization of 
overlapping masks. The resulting visualization scales cleanly to many ROIs and remains 
stable across time.

# Interpretation

Under this model, the rendered image should be interpreted as follows:
    - Brightness reflects the original imaging signal. 
    - Hue and saturation reflect ROI identity and activity. 
    - Color appears only when signal rises above baseline, reinforcing interpretability.

In effect, ROIs act as dynamic, spatially weighted color annotations layered onto a 
faithful grayscale rendering of the data. This makes the visualization suitable for 
dense, time-resolved imaging data where preserving intensity structure is essential.

# Mathematical formulation

The final RGB image is rendered as:
$$
RGB(t,y,x) = I(t,y,x)+\sum_{r}^{N}\Delta_r(t,y,x)
$$
for each roi $r$ for colormap $N$ where $I$ is the initial grayscale image,
and $\Delta_r$ is the chromatic deviation for each roi, defined as:
$$ 
\Delta_r(t,y,x) = a_r(t,y,x) * w_i(y, x) * g(t, y, x) * (C_r(t)-1)
$$
