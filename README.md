# CIM Lab 5: Forward models, null spaces and regularisation

This lab uses one complete Hydra experiment file, following the pattern of CIM Labs 3 and 4. The code is deliberately short: students change `conf/experiment.yaml` and inspect a small number of functions.

## Learning outcomes

By the end, you should be able to:

1. apply Gaussian, motion, defocus and Bayer forward models;
2. compose defocus and Bayer operators;
3. explain an approximate null space experimentally;
4. define first-order Tikhonov and total-variation priors;
5. compare regularised reconstructions;
6. generate `(measurement, target)` pairs for later learning experiments.

The first `uv sync` installs PyTorch and DeepInverse, so it may take a few minutes. With `device=auto`, the lab selects CUDA when available, otherwise Apple MPS on supported Macs, otherwise CPU. You can also set `device=cpu`, `device=cuda`, or `device=mps` explicitly.

## Image data

The default image is a small synthetic sample, so the first run works offline. Oxford-IIIT Pet is available for experiments with more images; on first use, `torchvision` downloads it into `data/oxford-iiit-pet`, then reuses those files. The full download is roughly 0.8 GB, so allow around 1 GB of disk space. `data.max_examples` limits how many images a run processes, not the initial download.

The Oxford-IIIT Pet dataset contains 7,349 cat and dog images. It is distributed under [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/); cite the [dataset page](https://www.robots.ox.ac.uk/~vgg/data/pets/) when redistributing images or derived materials.

For a fast run that does not download the full Pet dataset, use one built-in or local image. Select it with `data.name`; image names are filename stems in `data.root`, so `data.name=lib_left` loads `data/lib_left.jpg`. Supported file types are `.jpg`, `.jpeg`, and `.png`.

```powershell
uv run python -m cim_lab_5.main data.name=sample
uv run python -m cim_lab_5.main data.name=astronaut
uv run python -m cim_lab_5.main data.name=cameraman
uv run python -m cim_lab_5.main data.name=meeting_house
uv run python -m cim_lab_5.main data.name=lib_left
```

`sample` is synthetic and built into the code. Other named images are stored in `data/`; add your own image there and select it using its filename without the extension. The existing CelebA loader remains available with `data.name=celeba data.split=train`. For Pet images, set `data.name=pets`.

The bundled images are credited as follows:

- `astronaut.jpg`: NASA, [public domain image hosted by Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Bruce_McCandless_II_during_EVA_in_1984.jpg).
- `cameraman.jpg`: Davidlens, [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:Cameraman.jpg).
- `meeting_house.jpg`: Simon Carey, [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/), [Wikimedia Commons source](https://commons.wikimedia.org/wiki/File:Meeting_House,_University_of_Sussex_-_geograph.org.uk_-_3901696.jpg).
- `lib_left.jpg`: provided by Ivor Simpson.

## Part A: forward models

Run the default Gaussian experiment:

```powershell
uv run python -m cim_lab_5.main mode=forward experiment_name=gaussian_forward
```

Change `operator.name` to:

- `gaussian`
- `motion`
- `defocus`
- `bayer`
- `defocus_bayer`
- `downsample` (last resort only; super-resolution is next week)

A blur uses DeepInverse's `Blur` physics with an explicit point-spread function (PSF). Bayer sampling uses DeepInverse's `Demosaicing` mask, packed into a one-channel raw mosaic. The `downsample` option uses DeepInverse's `Downsampling` physics with no anti-aliasing filter, so it performs bare pixel decimation.

### Generate dataset pairs for learning models

Write some code in main.py to save out some synthetic observations and targets in a format that you can read back in again for training a future ML model
```powershell
uv run python -m cim_lab_5.main mode=dataset experiment_name=training_pairs
```

## Null-space challenge

> What is the largest change you can make to the target while keeping the measurement change below the tolerance?

Run:

```powershell
uv run python -m cim_lab_5.main mode=nullspace experiment_name=null_gaussian
```

The supplied search tries random high-frequency perturbations. Improve the score by changing only the values under `nullspace` in `conf/experiment.yaml`, then by designing a better perturbation in `highpass_noise()`.

Record both numbers:

$$
\|x'-x\|_{\mathrm{RMSE}}, \qquad \|A(x')-A(x)\|_{\mathrm{RMSE}}.
$$

Repeat for motion, defocus, Bayer and defocus+Bayer. Explain why the best hidden perturbation depends on the operator.

## Part B: regularised reconstruction

We reconstruct by minimising

$$
\hat{x}=\arg\min_x \frac{1}{N_y}\|Ax-y\|_2^2+\lambda R(x).
$$

### First-order Tikhonov

The lab uses a quadratic smoothness prior:

$$
R_{\mathrm{Tik}}(x)=\frac{1}{N_x}\|\nabla x\|_2^2.
$$

This strongly penalises large gradients and therefore prefers globally smooth images.

### Total variation

The isotropic TV prior is

$$
R_{\mathrm{TV}}(x)=\frac{1}{N_x}\sum_{\mathbf p}
\sqrt{(D_xx(\mathbf p))^2+(D_yx(\mathbf p))^2+\epsilon}.
$$

Unlike the squared Tikhonov penalty, TV can retain a smaller number of sharp edges.

### What is the adjoint?

The forward operator $A$ models how an image becomes a measurement. Its adjoint, $A^\ast$, maps a measurement back into image space and obeys

$$
\langle Ax,y\rangle=\langle x,A^\ast y\rangle
$$
for all images $x$ and measurements $y$. You can think of it as “back-projecting” measurement values into the image grid. It is useful for visualising measurements and as a reconstruction starting point, but it is **not generally an inverse**: blur and missing colour samples cannot simply be undone by applying the adjoint. For a concise mathematical definition, see [Adjoint (linear algebra)](https://en.wikipedia.org/wiki/Adjoint_(linear_algebra)).

### Further reading

- [Tikhonov regularization](https://en.wikipedia.org/wiki/Tikhonov_regularization) — the quadratic-penalty idea used here; this lab applies it to image gradients.
- Rudin, Osher and Fatemi, [“Nonlinear total variation based noise removal algorithms”](https://doi.org/10.1137/0727034) (1992) — the foundational paper behind total-variation image regularisation.

Read the "reconstruct" and "regulariser" function code in reconstruct.py to understand the optimisation procedure and regulariser implementation.
Run both on exactly the same observation:

```powershell
uv run python -m cim_lab_5.main mode=reconstruct reconstruction.method=tikhonov reconstruction.lambda=0.01 experiment_name=gaussian_tikhonov
uv run python -m cim_lab_5.main mode=reconstruct reconstruction.method=tv reconstruction.lambda=0.02 experiment_name=gaussian_tv
```

Try at least three values of `reconstruction.lambda` for one blur. Record under-regularised, useful and over-regularised cases. Then compare Gaussian, motion and defocus blur. Bayer reconstruction is intentionally difficult with these simple priors.

## Suggested experiments

1. **Forward-model gallery:** all five main operators on the same image.
2. **Null-space challenge:** maximise image RMSE under a fixed measurement RMSE tolerance.
3. **Regularisation sweep:** three `lambda` values for Tikhonov and TV.
4. **Model mismatch:** reconstruct motion-blurred data after changing the configured angle. What fails?
5. **Composite imaging:** compare Bayer alone with defocus+Bayer.

## Optional deblurring network pointer

Do not add another network to this lab. If you want to see a learned deblurring model, use the official DeepInverse examples and pretrained-model documentation after completing the classical experiments. The important question is: where is the prior represented in Tikhonov, TV and a trained network?
