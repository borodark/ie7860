Learning Processes


• The Widrow–Hoff rule (a.k.a. LMS or delta rule) updates weights to reduce mean squared error for a linear unit. For input x, output y = w·x, target d, error e = d - y, the update is:

  w := w + η e x

  It’s a gradient descent step on squared error, with learning rate η.

learning rate η

  - Learning rate η controls step size; too large causes divergence or oscillation, too small slows convergence.
  - Stability depends on the loss landscape; steep curvature demands smaller η to avoid overshooting.
  - For convex problems, η must be below a critical threshold tied to the Lipschitz constant of the gradient.
  - In nonconvex settings, large η can skip minima or amplify noise; small η can get stuck or crawl.
  - Adaptive schedules (decay, warmup, cosine) balance early speed with later stability.
  - Practical tuning: start moderate, monitor loss/gradient norms, and adjust based on instability signs.


```python
import keras
from keras import layers

model = keras.Sequential()
model.add(layers.Dense(64, kernel_initializer='uniform', input_shape=(10,)))
model.add(layers.Activation('softmax'))

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt)
```


## Boltzmann machines

(especially RBMs) have been applied to:

  - Unsupervised feature learning and dimensionality reduction
  - Collaborative filtering and recommender systems
  - Pretraining deep networks (historically, before modern optimizers)
  - Generative modeling of images, text, or binary data
  - Denoising and inpainting tasks
  - Anomaly detection by modeling typical data distributions

  Today they’re used less in mainstream production than VAEs/normalizing flows/diffusion models, but still show up in research and niche probabilistic
  modeling.
