

```python
#Since the warnings were not problematic, I suppressed them. 
import warnings
warnings.filterwarnings('ignore')
#standard imports
import numpy as np
import pandas as pd
import os
#matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import figure
#plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.figure_factory as ff
from  plotly  import __version__
#plotly offline
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
print(__version__) # requires version >= 1.9.0
init_notebook_mode(connected=True)
#scikitlearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sklearn
sklearn.__version__
from sklearn import datasets, linear_model
# Scientific libraries
from numpy import arange,array,ones
from scipy import stats
```

    3.1.0



<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



```python
df = pd.read_csv('~/Desktop/Python Exercises/featuresdf.csv')
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7qiZfU4dY1lWllzX7mPBI</td>
      <td>Shape of You</td>
      <td>Ed Sheeran</td>
      <td>0.825</td>
      <td>0.652</td>
      <td>1.0</td>
      <td>-3.183</td>
      <td>0.0</td>
      <td>0.0802</td>
      <td>0.581000</td>
      <td>0.000000</td>
      <td>0.0931</td>
      <td>0.9310</td>
      <td>95.977</td>
      <td>233713.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5CtI0qwDJkDQGwXD1H1cL</td>
      <td>Despacito - Remix</td>
      <td>Luis Fonsi</td>
      <td>0.694</td>
      <td>0.815</td>
      <td>2.0</td>
      <td>-4.328</td>
      <td>1.0</td>
      <td>0.1200</td>
      <td>0.229000</td>
      <td>0.000000</td>
      <td>0.0924</td>
      <td>0.8130</td>
      <td>88.931</td>
      <td>228827.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4aWmUDTfIPGksMNLV2rQP</td>
      <td>Despacito (Featuring Daddy Yankee)</td>
      <td>Luis Fonsi</td>
      <td>0.660</td>
      <td>0.786</td>
      <td>2.0</td>
      <td>-4.757</td>
      <td>1.0</td>
      <td>0.1700</td>
      <td>0.209000</td>
      <td>0.000000</td>
      <td>0.1120</td>
      <td>0.8460</td>
      <td>177.833</td>
      <td>228200.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6RUKPb4LETWmmr3iAEQkt</td>
      <td>Something Just Like This</td>
      <td>The Chainsmokers</td>
      <td>0.617</td>
      <td>0.635</td>
      <td>11.0</td>
      <td>-6.769</td>
      <td>0.0</td>
      <td>0.0317</td>
      <td>0.049800</td>
      <td>0.000014</td>
      <td>0.1640</td>
      <td>0.4460</td>
      <td>103.019</td>
      <td>247160.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3DXncPQOG4VBw3QHh3S81</td>
      <td>I'm the One</td>
      <td>DJ Khaled</td>
      <td>0.609</td>
      <td>0.668</td>
      <td>7.0</td>
      <td>-4.284</td>
      <td>1.0</td>
      <td>0.0367</td>
      <td>0.055200</td>
      <td>0.000000</td>
      <td>0.1670</td>
      <td>0.8110</td>
      <td>80.924</td>
      <td>288600.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7KXjTSCq5nL1LoYtL7XAw</td>
      <td>HUMBLE.</td>
      <td>Kendrick Lamar</td>
      <td>0.904</td>
      <td>0.611</td>
      <td>1.0</td>
      <td>-6.842</td>
      <td>0.0</td>
      <td>0.0888</td>
      <td>0.000259</td>
      <td>0.000020</td>
      <td>0.0976</td>
      <td>0.4000</td>
      <td>150.020</td>
      <td>177000.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3eR23VReFzcdmS7TYCrhC</td>
      <td>It Ain't Me (with Selena Gomez)</td>
      <td>Kygo</td>
      <td>0.640</td>
      <td>0.533</td>
      <td>0.0</td>
      <td>-6.596</td>
      <td>1.0</td>
      <td>0.0706</td>
      <td>0.119000</td>
      <td>0.000000</td>
      <td>0.0864</td>
      <td>0.5150</td>
      <td>99.968</td>
      <td>220781.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3B54sVLJ402zGa6Xm4YGN</td>
      <td>Unforgettable</td>
      <td>French Montana</td>
      <td>0.726</td>
      <td>0.769</td>
      <td>6.0</td>
      <td>-5.043</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.029300</td>
      <td>0.010100</td>
      <td>0.1040</td>
      <td>0.7330</td>
      <td>97.985</td>
      <td>233902.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0KKkJNfGyhkQ5aFogxQAP</td>
      <td>That's What I Like</td>
      <td>Bruno Mars</td>
      <td>0.853</td>
      <td>0.560</td>
      <td>1.0</td>
      <td>-4.961</td>
      <td>1.0</td>
      <td>0.0406</td>
      <td>0.013000</td>
      <td>0.000000</td>
      <td>0.0944</td>
      <td>0.8600</td>
      <td>134.066</td>
      <td>206693.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3NdDpSvN911VPGivFlV5d</td>
      <td>I Don’t Wanna Live Forever (Fifty Shades Darke...</td>
      <td>ZAYN</td>
      <td>0.735</td>
      <td>0.451</td>
      <td>0.0</td>
      <td>-8.374</td>
      <td>1.0</td>
      <td>0.0585</td>
      <td>0.063100</td>
      <td>0.000013</td>
      <td>0.3250</td>
      <td>0.0862</td>
      <td>117.973</td>
      <td>245200.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7GX5flRQZVHRAGd6B4TmD</td>
      <td>XO TOUR Llif3</td>
      <td>Lil Uzi Vert</td>
      <td>0.732</td>
      <td>0.750</td>
      <td>11.0</td>
      <td>-6.366</td>
      <td>0.0</td>
      <td>0.2310</td>
      <td>0.002640</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.4010</td>
      <td>155.096</td>
      <td>182707.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>72jbDTw1piOOj770jWNea</td>
      <td>Paris</td>
      <td>The Chainsmokers</td>
      <td>0.653</td>
      <td>0.658</td>
      <td>2.0</td>
      <td>-6.428</td>
      <td>1.0</td>
      <td>0.0304</td>
      <td>0.021500</td>
      <td>0.000002</td>
      <td>0.0939</td>
      <td>0.2190</td>
      <td>99.990</td>
      <td>221507.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0dA2Mk56wEzDgegdC6R17</td>
      <td>Stay (with Alessia Cara)</td>
      <td>Zedd</td>
      <td>0.679</td>
      <td>0.634</td>
      <td>5.0</td>
      <td>-5.024</td>
      <td>0.0</td>
      <td>0.0654</td>
      <td>0.232000</td>
      <td>0.000000</td>
      <td>0.1150</td>
      <td>0.4980</td>
      <td>102.013</td>
      <td>210091.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4iLqG9SeJSnt0cSPICSjx</td>
      <td>Attention</td>
      <td>Charlie Puth</td>
      <td>0.774</td>
      <td>0.626</td>
      <td>3.0</td>
      <td>-4.432</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.096900</td>
      <td>0.000031</td>
      <td>0.0848</td>
      <td>0.7770</td>
      <td>100.041</td>
      <td>211475.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0VgkVdmE4gld66l8iyGjg</td>
      <td>Mask Off</td>
      <td>Future</td>
      <td>0.833</td>
      <td>0.434</td>
      <td>2.0</td>
      <td>-8.795</td>
      <td>1.0</td>
      <td>0.4310</td>
      <td>0.010200</td>
      <td>0.021900</td>
      <td>0.1650</td>
      <td>0.2810</td>
      <td>150.062</td>
      <td>204600.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3a1lNhkSLSkpJE4MSHpDu</td>
      <td>Congratulations</td>
      <td>Post Malone</td>
      <td>0.627</td>
      <td>0.812</td>
      <td>6.0</td>
      <td>-4.215</td>
      <td>1.0</td>
      <td>0.0358</td>
      <td>0.198000</td>
      <td>0.000000</td>
      <td>0.2120</td>
      <td>0.5040</td>
      <td>123.071</td>
      <td>220293.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6kex4EBAj0WHXDKZMEJaa</td>
      <td>Swalla (feat. Nicki Minaj &amp; Ty Dolla $ign)</td>
      <td>Jason Derulo</td>
      <td>0.696</td>
      <td>0.817</td>
      <td>1.0</td>
      <td>-3.862</td>
      <td>1.0</td>
      <td>0.1090</td>
      <td>0.075000</td>
      <td>0.000000</td>
      <td>0.1870</td>
      <td>0.7820</td>
      <td>98.064</td>
      <td>216409.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6PCUP3dWmTjcTtXY02oFd</td>
      <td>Castle on the Hill</td>
      <td>Ed Sheeran</td>
      <td>0.461</td>
      <td>0.834</td>
      <td>2.0</td>
      <td>-4.868</td>
      <td>1.0</td>
      <td>0.0989</td>
      <td>0.023200</td>
      <td>0.000011</td>
      <td>0.1400</td>
      <td>0.4710</td>
      <td>135.007</td>
      <td>261154.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5knuzwU65gJK7IF5yJsua</td>
      <td>Rockabye (feat. Sean Paul &amp; Anne-Marie)</td>
      <td>Clean Bandit</td>
      <td>0.720</td>
      <td>0.763</td>
      <td>9.0</td>
      <td>-4.068</td>
      <td>0.0</td>
      <td>0.0523</td>
      <td>0.406000</td>
      <td>0.000000</td>
      <td>0.1800</td>
      <td>0.7420</td>
      <td>101.965</td>
      <td>251088.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0CcQNd8CINkwQfe1RDtGV</td>
      <td>Believer</td>
      <td>Imagine Dragons</td>
      <td>0.779</td>
      <td>0.787</td>
      <td>10.0</td>
      <td>-4.305</td>
      <td>0.0</td>
      <td>0.1080</td>
      <td>0.052400</td>
      <td>0.000000</td>
      <td>0.1400</td>
      <td>0.7080</td>
      <td>124.982</td>
      <td>204347.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2rb5MvYT7ZIxbKW5hfcHx</td>
      <td>Mi Gente</td>
      <td>J Balvin</td>
      <td>0.543</td>
      <td>0.677</td>
      <td>11.0</td>
      <td>-4.915</td>
      <td>0.0</td>
      <td>0.0993</td>
      <td>0.014800</td>
      <td>0.000006</td>
      <td>0.1300</td>
      <td>0.2940</td>
      <td>103.809</td>
      <td>189440.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0tKcYR2II1VCQWT79i5Nr</td>
      <td>Thunder</td>
      <td>Imagine Dragons</td>
      <td>0.600</td>
      <td>0.810</td>
      <td>0.0</td>
      <td>-4.749</td>
      <td>1.0</td>
      <td>0.0479</td>
      <td>0.006830</td>
      <td>0.210000</td>
      <td>0.1550</td>
      <td>0.2980</td>
      <td>167.880</td>
      <td>187147.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5uCax9HTNlzGybIStD3vD</td>
      <td>Say You Won't Let Go</td>
      <td>James Arthur</td>
      <td>0.358</td>
      <td>0.557</td>
      <td>10.0</td>
      <td>-7.398</td>
      <td>1.0</td>
      <td>0.0590</td>
      <td>0.695000</td>
      <td>0.000000</td>
      <td>0.0902</td>
      <td>0.4940</td>
      <td>85.043</td>
      <td>211467.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>79cuOz3SPQTuFrp8WgftA</td>
      <td>There's Nothing Holdin' Me Back</td>
      <td>Shawn Mendes</td>
      <td>0.857</td>
      <td>0.800</td>
      <td>2.0</td>
      <td>-4.035</td>
      <td>1.0</td>
      <td>0.0583</td>
      <td>0.381000</td>
      <td>0.000000</td>
      <td>0.0913</td>
      <td>0.9660</td>
      <td>121.996</td>
      <td>199440.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6De0lHrwBfPfrhorm9q1X</td>
      <td>Me Rehúso</td>
      <td>Danny Ocean</td>
      <td>0.744</td>
      <td>0.804</td>
      <td>1.0</td>
      <td>-6.327</td>
      <td>1.0</td>
      <td>0.0677</td>
      <td>0.023100</td>
      <td>0.000000</td>
      <td>0.0494</td>
      <td>0.4260</td>
      <td>104.823</td>
      <td>205715.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6D0b04NJIKfEMg040WioJ</td>
      <td>Issues</td>
      <td>Julia Michaels</td>
      <td>0.706</td>
      <td>0.427</td>
      <td>8.0</td>
      <td>-6.864</td>
      <td>1.0</td>
      <td>0.0879</td>
      <td>0.413000</td>
      <td>0.000000</td>
      <td>0.0609</td>
      <td>0.4200</td>
      <td>113.804</td>
      <td>176320.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0afhq8XCExXpqazXczTSv</td>
      <td>Galway Girl</td>
      <td>Ed Sheeran</td>
      <td>0.624</td>
      <td>0.876</td>
      <td>9.0</td>
      <td>-3.374</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.073500</td>
      <td>0.000000</td>
      <td>0.3270</td>
      <td>0.7810</td>
      <td>99.943</td>
      <td>170827.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3ebXMykcMXOcLeJ9xZ17X</td>
      <td>Scared to Be Lonely</td>
      <td>Martin Garrix</td>
      <td>0.584</td>
      <td>0.540</td>
      <td>1.0</td>
      <td>-7.786</td>
      <td>0.0</td>
      <td>0.0576</td>
      <td>0.089500</td>
      <td>0.000000</td>
      <td>0.2610</td>
      <td>0.1950</td>
      <td>137.972</td>
      <td>220883.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7BKLCZ1jbUBVqRi2FVlTV</td>
      <td>Closer</td>
      <td>The Chainsmokers</td>
      <td>0.748</td>
      <td>0.524</td>
      <td>8.0</td>
      <td>-5.599</td>
      <td>1.0</td>
      <td>0.0338</td>
      <td>0.414000</td>
      <td>0.000000</td>
      <td>0.1110</td>
      <td>0.6610</td>
      <td>95.010</td>
      <td>244960.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1x5sYLZiu9r5E43kMlt9f</td>
      <td>Symphony (feat. Zara Larsson)</td>
      <td>Clean Bandit</td>
      <td>0.707</td>
      <td>0.629</td>
      <td>0.0</td>
      <td>-4.581</td>
      <td>0.0</td>
      <td>0.0563</td>
      <td>0.259000</td>
      <td>0.000016</td>
      <td>0.1380</td>
      <td>0.4570</td>
      <td>122.863</td>
      <td>212459.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5Ohxk2dO5COHF1krpoPig</td>
      <td>Sign of the Times</td>
      <td>Harry Styles</td>
      <td>0.516</td>
      <td>0.595</td>
      <td>5.0</td>
      <td>-4.630</td>
      <td>1.0</td>
      <td>0.0313</td>
      <td>0.027500</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.2220</td>
      <td>119.972</td>
      <td>340707.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>6gBFPUFcJLzWGx4lenP6h</td>
      <td>goosebumps</td>
      <td>Travis Scott</td>
      <td>0.841</td>
      <td>0.728</td>
      <td>7.0</td>
      <td>-3.370</td>
      <td>1.0</td>
      <td>0.0484</td>
      <td>0.084700</td>
      <td>0.000000</td>
      <td>0.1490</td>
      <td>0.4300</td>
      <td>130.049</td>
      <td>243837.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>5Z3GHaZ6ec9bsiI5Benrb</td>
      <td>Young Dumb &amp; Broke</td>
      <td>Khalid</td>
      <td>0.798</td>
      <td>0.539</td>
      <td>1.0</td>
      <td>-6.351</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.199000</td>
      <td>0.000017</td>
      <td>0.1650</td>
      <td>0.3940</td>
      <td>136.949</td>
      <td>202547.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>6jA8HL9i4QGzsj6fjoxp8</td>
      <td>There for You</td>
      <td>Martin Garrix</td>
      <td>0.611</td>
      <td>0.644</td>
      <td>6.0</td>
      <td>-7.607</td>
      <td>0.0</td>
      <td>0.0553</td>
      <td>0.124000</td>
      <td>0.000000</td>
      <td>0.1240</td>
      <td>0.1300</td>
      <td>105.969</td>
      <td>221904.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>21TdkDRXuAB3k90ujRU1e</td>
      <td>Cold (feat. Future)</td>
      <td>Maroon 5</td>
      <td>0.697</td>
      <td>0.716</td>
      <td>9.0</td>
      <td>-6.288</td>
      <td>0.0</td>
      <td>0.1130</td>
      <td>0.118000</td>
      <td>0.000000</td>
      <td>0.0424</td>
      <td>0.5060</td>
      <td>99.905</td>
      <td>234308.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>7vGuf3Y35N4wmASOKLUVV</td>
      <td>Silence</td>
      <td>Marshmello</td>
      <td>0.520</td>
      <td>0.761</td>
      <td>4.0</td>
      <td>-3.093</td>
      <td>1.0</td>
      <td>0.0853</td>
      <td>0.256000</td>
      <td>0.000005</td>
      <td>0.1700</td>
      <td>0.2860</td>
      <td>141.971</td>
      <td>180823.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1mXVgsBdtIVeCLJnSnmtd</td>
      <td>Too Good At Goodbyes</td>
      <td>Sam Smith</td>
      <td>0.698</td>
      <td>0.375</td>
      <td>5.0</td>
      <td>-8.279</td>
      <td>1.0</td>
      <td>0.0491</td>
      <td>0.652000</td>
      <td>0.000000</td>
      <td>0.1730</td>
      <td>0.5340</td>
      <td>91.920</td>
      <td>201000.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3EmmCZoqpWOTY1g2GBwJo</td>
      <td>Just Hold On</td>
      <td>Steve Aoki</td>
      <td>0.647</td>
      <td>0.932</td>
      <td>11.0</td>
      <td>-3.515</td>
      <td>1.0</td>
      <td>0.0824</td>
      <td>0.003830</td>
      <td>0.000002</td>
      <td>0.0574</td>
      <td>0.3740</td>
      <td>114.991</td>
      <td>198774.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6uFsE1JgZ20EXyU0JQZbU</td>
      <td>Look What You Made Me Do</td>
      <td>Taylor Swift</td>
      <td>0.773</td>
      <td>0.680</td>
      <td>9.0</td>
      <td>-6.378</td>
      <td>0.0</td>
      <td>0.1410</td>
      <td>0.213000</td>
      <td>0.000016</td>
      <td>0.1220</td>
      <td>0.4970</td>
      <td>128.062</td>
      <td>211859.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0CokSRCu5hZgPxcZBaEzV</td>
      <td>Glorious (feat. Skylar Grey)</td>
      <td>Macklemore</td>
      <td>0.731</td>
      <td>0.794</td>
      <td>0.0</td>
      <td>-5.126</td>
      <td>0.0</td>
      <td>0.0522</td>
      <td>0.032300</td>
      <td>0.000026</td>
      <td>0.1120</td>
      <td>0.3560</td>
      <td>139.994</td>
      <td>220454.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>6875MeXyCW0wLyT72Eetm</td>
      <td>Starving</td>
      <td>Hailee Steinfeld</td>
      <td>0.721</td>
      <td>0.626</td>
      <td>4.0</td>
      <td>-4.200</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.402000</td>
      <td>0.000000</td>
      <td>0.1020</td>
      <td>0.5580</td>
      <td>99.914</td>
      <td>181933.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>3AEZUABDXNtecAOSC1qTf</td>
      <td>Reggaetón Lento (Bailemos)</td>
      <td>CNCO</td>
      <td>0.761</td>
      <td>0.838</td>
      <td>4.0</td>
      <td>-3.073</td>
      <td>0.0</td>
      <td>0.0502</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.1760</td>
      <td>0.7100</td>
      <td>93.974</td>
      <td>222560.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>3E2Zh20GDCR9B1EYjfXWy</td>
      <td>Weak</td>
      <td>AJR</td>
      <td>0.673</td>
      <td>0.637</td>
      <td>5.0</td>
      <td>-4.518</td>
      <td>1.0</td>
      <td>0.0429</td>
      <td>0.137000</td>
      <td>0.000000</td>
      <td>0.1840</td>
      <td>0.6780</td>
      <td>123.980</td>
      <td>201160.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>4pLwZjInHj3SimIyN9SnO</td>
      <td>Side To Side</td>
      <td>Ariana Grande</td>
      <td>0.648</td>
      <td>0.738</td>
      <td>6.0</td>
      <td>-5.883</td>
      <td>0.0</td>
      <td>0.2470</td>
      <td>0.040800</td>
      <td>0.000000</td>
      <td>0.2920</td>
      <td>0.6030</td>
      <td>159.145</td>
      <td>226160.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>3QwBODjSEzelZyVjxPOHd</td>
      <td>Otra Vez (feat. J Balvin)</td>
      <td>Zion &amp; Lennox</td>
      <td>0.832</td>
      <td>0.772</td>
      <td>10.0</td>
      <td>-5.429</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.055900</td>
      <td>0.000486</td>
      <td>0.4400</td>
      <td>0.7040</td>
      <td>96.016</td>
      <td>209453.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1wjzFQodRWrPcQ0AnYnvQ</td>
      <td>I Like Me Better</td>
      <td>Lauv</td>
      <td>0.752</td>
      <td>0.505</td>
      <td>9.0</td>
      <td>-7.621</td>
      <td>1.0</td>
      <td>0.2530</td>
      <td>0.535000</td>
      <td>0.000003</td>
      <td>0.1040</td>
      <td>0.4190</td>
      <td>91.970</td>
      <td>197437.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>04DwTuZ2VBdJCCC5TROn7</td>
      <td>In the Name of Love</td>
      <td>Martin Garrix</td>
      <td>0.490</td>
      <td>0.485</td>
      <td>4.0</td>
      <td>-6.237</td>
      <td>0.0</td>
      <td>0.0406</td>
      <td>0.059200</td>
      <td>0.000000</td>
      <td>0.3370</td>
      <td>0.1960</td>
      <td>133.889</td>
      <td>195840.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6DNtNfH8hXkqOX1sjqmI7</td>
      <td>Cold Water (feat. Justin Bieber &amp; MØ)</td>
      <td>Major Lazer</td>
      <td>0.608</td>
      <td>0.798</td>
      <td>6.0</td>
      <td>-5.092</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.073600</td>
      <td>0.000000</td>
      <td>0.1560</td>
      <td>0.5010</td>
      <td>92.943</td>
      <td>185352.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1UZOjK1BwmwWU14Erba9C</td>
      <td>Malibu</td>
      <td>Miley Cyrus</td>
      <td>0.573</td>
      <td>0.781</td>
      <td>8.0</td>
      <td>-6.406</td>
      <td>1.0</td>
      <td>0.0555</td>
      <td>0.076700</td>
      <td>0.000026</td>
      <td>0.0813</td>
      <td>0.3430</td>
      <td>139.934</td>
      <td>231907.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4b4KcovePX8Ke2cLIQTLM</td>
      <td>All Night</td>
      <td>The Vamps</td>
      <td>0.544</td>
      <td>0.809</td>
      <td>8.0</td>
      <td>-5.098</td>
      <td>1.0</td>
      <td>0.0363</td>
      <td>0.003800</td>
      <td>0.000000</td>
      <td>0.3230</td>
      <td>0.4480</td>
      <td>145.017</td>
      <td>197640.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1a5Yu5L18qNxVhXx38njO</td>
      <td>Hear Me Now</td>
      <td>Alok</td>
      <td>0.789</td>
      <td>0.442</td>
      <td>11.0</td>
      <td>-7.844</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.586000</td>
      <td>0.003660</td>
      <td>0.0927</td>
      <td>0.4500</td>
      <td>121.971</td>
      <td>192846.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4c2W3VKsOFoIg2SFaO6DY</td>
      <td>Your Song</td>
      <td>Rita Ora</td>
      <td>0.855</td>
      <td>0.624</td>
      <td>1.0</td>
      <td>-4.093</td>
      <td>1.0</td>
      <td>0.0488</td>
      <td>0.158000</td>
      <td>0.000000</td>
      <td>0.0513</td>
      <td>0.9620</td>
      <td>117.959</td>
      <td>180757.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>22eADXu8DfOAUEDw4vU8q</td>
      <td>Ahora Dice</td>
      <td>Chris Jeday</td>
      <td>0.708</td>
      <td>0.693</td>
      <td>6.0</td>
      <td>-5.516</td>
      <td>1.0</td>
      <td>0.1380</td>
      <td>0.246000</td>
      <td>0.000000</td>
      <td>0.1290</td>
      <td>0.4270</td>
      <td>143.965</td>
      <td>271080.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>7nZmah2llfvLDiUjm0kiy</td>
      <td>Friends (with BloodPop®)</td>
      <td>Justin Bieber</td>
      <td>0.744</td>
      <td>0.739</td>
      <td>8.0</td>
      <td>-5.350</td>
      <td>1.0</td>
      <td>0.0387</td>
      <td>0.004590</td>
      <td>0.000000</td>
      <td>0.3060</td>
      <td>0.6490</td>
      <td>104.990</td>
      <td>189467.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2fQrGHiQOvpL9UgPvtYy6</td>
      <td>Bank Account</td>
      <td>21 Savage</td>
      <td>0.884</td>
      <td>0.346</td>
      <td>8.0</td>
      <td>-8.228</td>
      <td>0.0</td>
      <td>0.3510</td>
      <td>0.015100</td>
      <td>0.000007</td>
      <td>0.0871</td>
      <td>0.3760</td>
      <td>75.016</td>
      <td>220307.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1PSBzsahR2AKwLJgx8ehB</td>
      <td>Bad Things (with Camila Cabello)</td>
      <td>Machine Gun Kelly</td>
      <td>0.675</td>
      <td>0.690</td>
      <td>2.0</td>
      <td>-4.761</td>
      <td>1.0</td>
      <td>0.1320</td>
      <td>0.210000</td>
      <td>0.000000</td>
      <td>0.2870</td>
      <td>0.2720</td>
      <td>137.817</td>
      <td>239293.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0QsvXIfqM0zZoerQfsI9l</td>
      <td>Don't Let Me Down</td>
      <td>The Chainsmokers</td>
      <td>0.542</td>
      <td>0.859</td>
      <td>11.0</td>
      <td>-5.651</td>
      <td>1.0</td>
      <td>0.1970</td>
      <td>0.160000</td>
      <td>0.004660</td>
      <td>0.1370</td>
      <td>0.4030</td>
      <td>159.797</td>
      <td>208053.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>7mldq42yDuxiUNn08nvzH</td>
      <td>Body Like A Back Road</td>
      <td>Sam Hunt</td>
      <td>0.731</td>
      <td>0.469</td>
      <td>5.0</td>
      <td>-7.226</td>
      <td>1.0</td>
      <td>0.0326</td>
      <td>0.463000</td>
      <td>0.000001</td>
      <td>0.1030</td>
      <td>0.6310</td>
      <td>98.963</td>
      <td>165387.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>7i2DJ88J7jQ8K7zqFX2fW</td>
      <td>Now Or Never</td>
      <td>Halsey</td>
      <td>0.658</td>
      <td>0.588</td>
      <td>6.0</td>
      <td>-4.902</td>
      <td>0.0</td>
      <td>0.0367</td>
      <td>0.105000</td>
      <td>0.000001</td>
      <td>0.1250</td>
      <td>0.4340</td>
      <td>110.075</td>
      <td>214802.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1j4kHkkpqZRBwE0A4CN4Y</td>
      <td>Dusk Till Dawn - Radio Edit</td>
      <td>ZAYN</td>
      <td>0.258</td>
      <td>0.437</td>
      <td>11.0</td>
      <td>-6.593</td>
      <td>0.0</td>
      <td>0.0390</td>
      <td>0.101000</td>
      <td>0.000001</td>
      <td>0.1060</td>
      <td>0.0967</td>
      <td>180.043</td>
      <td>239000.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 16 columns</p>
</div>




```python
df_transform = pd.read_csv('~/Desktop/Python Exercises/featuresdf.csv')
df_transform
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7qiZfU4dY1lWllzX7mPBI</td>
      <td>Shape of You</td>
      <td>Ed Sheeran</td>
      <td>0.825</td>
      <td>0.652</td>
      <td>1.0</td>
      <td>-3.183</td>
      <td>0.0</td>
      <td>0.0802</td>
      <td>0.581000</td>
      <td>0.000000</td>
      <td>0.0931</td>
      <td>0.9310</td>
      <td>95.977</td>
      <td>233713.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5CtI0qwDJkDQGwXD1H1cL</td>
      <td>Despacito - Remix</td>
      <td>Luis Fonsi</td>
      <td>0.694</td>
      <td>0.815</td>
      <td>2.0</td>
      <td>-4.328</td>
      <td>1.0</td>
      <td>0.1200</td>
      <td>0.229000</td>
      <td>0.000000</td>
      <td>0.0924</td>
      <td>0.8130</td>
      <td>88.931</td>
      <td>228827.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4aWmUDTfIPGksMNLV2rQP</td>
      <td>Despacito (Featuring Daddy Yankee)</td>
      <td>Luis Fonsi</td>
      <td>0.660</td>
      <td>0.786</td>
      <td>2.0</td>
      <td>-4.757</td>
      <td>1.0</td>
      <td>0.1700</td>
      <td>0.209000</td>
      <td>0.000000</td>
      <td>0.1120</td>
      <td>0.8460</td>
      <td>177.833</td>
      <td>228200.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6RUKPb4LETWmmr3iAEQkt</td>
      <td>Something Just Like This</td>
      <td>The Chainsmokers</td>
      <td>0.617</td>
      <td>0.635</td>
      <td>11.0</td>
      <td>-6.769</td>
      <td>0.0</td>
      <td>0.0317</td>
      <td>0.049800</td>
      <td>0.000014</td>
      <td>0.1640</td>
      <td>0.4460</td>
      <td>103.019</td>
      <td>247160.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3DXncPQOG4VBw3QHh3S81</td>
      <td>I'm the One</td>
      <td>DJ Khaled</td>
      <td>0.609</td>
      <td>0.668</td>
      <td>7.0</td>
      <td>-4.284</td>
      <td>1.0</td>
      <td>0.0367</td>
      <td>0.055200</td>
      <td>0.000000</td>
      <td>0.1670</td>
      <td>0.8110</td>
      <td>80.924</td>
      <td>288600.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7KXjTSCq5nL1LoYtL7XAw</td>
      <td>HUMBLE.</td>
      <td>Kendrick Lamar</td>
      <td>0.904</td>
      <td>0.611</td>
      <td>1.0</td>
      <td>-6.842</td>
      <td>0.0</td>
      <td>0.0888</td>
      <td>0.000259</td>
      <td>0.000020</td>
      <td>0.0976</td>
      <td>0.4000</td>
      <td>150.020</td>
      <td>177000.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3eR23VReFzcdmS7TYCrhC</td>
      <td>It Ain't Me (with Selena Gomez)</td>
      <td>Kygo</td>
      <td>0.640</td>
      <td>0.533</td>
      <td>0.0</td>
      <td>-6.596</td>
      <td>1.0</td>
      <td>0.0706</td>
      <td>0.119000</td>
      <td>0.000000</td>
      <td>0.0864</td>
      <td>0.5150</td>
      <td>99.968</td>
      <td>220781.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3B54sVLJ402zGa6Xm4YGN</td>
      <td>Unforgettable</td>
      <td>French Montana</td>
      <td>0.726</td>
      <td>0.769</td>
      <td>6.0</td>
      <td>-5.043</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.029300</td>
      <td>0.010100</td>
      <td>0.1040</td>
      <td>0.7330</td>
      <td>97.985</td>
      <td>233902.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0KKkJNfGyhkQ5aFogxQAP</td>
      <td>That's What I Like</td>
      <td>Bruno Mars</td>
      <td>0.853</td>
      <td>0.560</td>
      <td>1.0</td>
      <td>-4.961</td>
      <td>1.0</td>
      <td>0.0406</td>
      <td>0.013000</td>
      <td>0.000000</td>
      <td>0.0944</td>
      <td>0.8600</td>
      <td>134.066</td>
      <td>206693.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3NdDpSvN911VPGivFlV5d</td>
      <td>I Don’t Wanna Live Forever (Fifty Shades Darke...</td>
      <td>ZAYN</td>
      <td>0.735</td>
      <td>0.451</td>
      <td>0.0</td>
      <td>-8.374</td>
      <td>1.0</td>
      <td>0.0585</td>
      <td>0.063100</td>
      <td>0.000013</td>
      <td>0.3250</td>
      <td>0.0862</td>
      <td>117.973</td>
      <td>245200.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7GX5flRQZVHRAGd6B4TmD</td>
      <td>XO TOUR Llif3</td>
      <td>Lil Uzi Vert</td>
      <td>0.732</td>
      <td>0.750</td>
      <td>11.0</td>
      <td>-6.366</td>
      <td>0.0</td>
      <td>0.2310</td>
      <td>0.002640</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.4010</td>
      <td>155.096</td>
      <td>182707.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>72jbDTw1piOOj770jWNea</td>
      <td>Paris</td>
      <td>The Chainsmokers</td>
      <td>0.653</td>
      <td>0.658</td>
      <td>2.0</td>
      <td>-6.428</td>
      <td>1.0</td>
      <td>0.0304</td>
      <td>0.021500</td>
      <td>0.000002</td>
      <td>0.0939</td>
      <td>0.2190</td>
      <td>99.990</td>
      <td>221507.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0dA2Mk56wEzDgegdC6R17</td>
      <td>Stay (with Alessia Cara)</td>
      <td>Zedd</td>
      <td>0.679</td>
      <td>0.634</td>
      <td>5.0</td>
      <td>-5.024</td>
      <td>0.0</td>
      <td>0.0654</td>
      <td>0.232000</td>
      <td>0.000000</td>
      <td>0.1150</td>
      <td>0.4980</td>
      <td>102.013</td>
      <td>210091.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4iLqG9SeJSnt0cSPICSjx</td>
      <td>Attention</td>
      <td>Charlie Puth</td>
      <td>0.774</td>
      <td>0.626</td>
      <td>3.0</td>
      <td>-4.432</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.096900</td>
      <td>0.000031</td>
      <td>0.0848</td>
      <td>0.7770</td>
      <td>100.041</td>
      <td>211475.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0VgkVdmE4gld66l8iyGjg</td>
      <td>Mask Off</td>
      <td>Future</td>
      <td>0.833</td>
      <td>0.434</td>
      <td>2.0</td>
      <td>-8.795</td>
      <td>1.0</td>
      <td>0.4310</td>
      <td>0.010200</td>
      <td>0.021900</td>
      <td>0.1650</td>
      <td>0.2810</td>
      <td>150.062</td>
      <td>204600.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3a1lNhkSLSkpJE4MSHpDu</td>
      <td>Congratulations</td>
      <td>Post Malone</td>
      <td>0.627</td>
      <td>0.812</td>
      <td>6.0</td>
      <td>-4.215</td>
      <td>1.0</td>
      <td>0.0358</td>
      <td>0.198000</td>
      <td>0.000000</td>
      <td>0.2120</td>
      <td>0.5040</td>
      <td>123.071</td>
      <td>220293.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6kex4EBAj0WHXDKZMEJaa</td>
      <td>Swalla (feat. Nicki Minaj &amp; Ty Dolla $ign)</td>
      <td>Jason Derulo</td>
      <td>0.696</td>
      <td>0.817</td>
      <td>1.0</td>
      <td>-3.862</td>
      <td>1.0</td>
      <td>0.1090</td>
      <td>0.075000</td>
      <td>0.000000</td>
      <td>0.1870</td>
      <td>0.7820</td>
      <td>98.064</td>
      <td>216409.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6PCUP3dWmTjcTtXY02oFd</td>
      <td>Castle on the Hill</td>
      <td>Ed Sheeran</td>
      <td>0.461</td>
      <td>0.834</td>
      <td>2.0</td>
      <td>-4.868</td>
      <td>1.0</td>
      <td>0.0989</td>
      <td>0.023200</td>
      <td>0.000011</td>
      <td>0.1400</td>
      <td>0.4710</td>
      <td>135.007</td>
      <td>261154.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5knuzwU65gJK7IF5yJsua</td>
      <td>Rockabye (feat. Sean Paul &amp; Anne-Marie)</td>
      <td>Clean Bandit</td>
      <td>0.720</td>
      <td>0.763</td>
      <td>9.0</td>
      <td>-4.068</td>
      <td>0.0</td>
      <td>0.0523</td>
      <td>0.406000</td>
      <td>0.000000</td>
      <td>0.1800</td>
      <td>0.7420</td>
      <td>101.965</td>
      <td>251088.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0CcQNd8CINkwQfe1RDtGV</td>
      <td>Believer</td>
      <td>Imagine Dragons</td>
      <td>0.779</td>
      <td>0.787</td>
      <td>10.0</td>
      <td>-4.305</td>
      <td>0.0</td>
      <td>0.1080</td>
      <td>0.052400</td>
      <td>0.000000</td>
      <td>0.1400</td>
      <td>0.7080</td>
      <td>124.982</td>
      <td>204347.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2rb5MvYT7ZIxbKW5hfcHx</td>
      <td>Mi Gente</td>
      <td>J Balvin</td>
      <td>0.543</td>
      <td>0.677</td>
      <td>11.0</td>
      <td>-4.915</td>
      <td>0.0</td>
      <td>0.0993</td>
      <td>0.014800</td>
      <td>0.000006</td>
      <td>0.1300</td>
      <td>0.2940</td>
      <td>103.809</td>
      <td>189440.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0tKcYR2II1VCQWT79i5Nr</td>
      <td>Thunder</td>
      <td>Imagine Dragons</td>
      <td>0.600</td>
      <td>0.810</td>
      <td>0.0</td>
      <td>-4.749</td>
      <td>1.0</td>
      <td>0.0479</td>
      <td>0.006830</td>
      <td>0.210000</td>
      <td>0.1550</td>
      <td>0.2980</td>
      <td>167.880</td>
      <td>187147.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5uCax9HTNlzGybIStD3vD</td>
      <td>Say You Won't Let Go</td>
      <td>James Arthur</td>
      <td>0.358</td>
      <td>0.557</td>
      <td>10.0</td>
      <td>-7.398</td>
      <td>1.0</td>
      <td>0.0590</td>
      <td>0.695000</td>
      <td>0.000000</td>
      <td>0.0902</td>
      <td>0.4940</td>
      <td>85.043</td>
      <td>211467.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>79cuOz3SPQTuFrp8WgftA</td>
      <td>There's Nothing Holdin' Me Back</td>
      <td>Shawn Mendes</td>
      <td>0.857</td>
      <td>0.800</td>
      <td>2.0</td>
      <td>-4.035</td>
      <td>1.0</td>
      <td>0.0583</td>
      <td>0.381000</td>
      <td>0.000000</td>
      <td>0.0913</td>
      <td>0.9660</td>
      <td>121.996</td>
      <td>199440.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6De0lHrwBfPfrhorm9q1X</td>
      <td>Me Rehúso</td>
      <td>Danny Ocean</td>
      <td>0.744</td>
      <td>0.804</td>
      <td>1.0</td>
      <td>-6.327</td>
      <td>1.0</td>
      <td>0.0677</td>
      <td>0.023100</td>
      <td>0.000000</td>
      <td>0.0494</td>
      <td>0.4260</td>
      <td>104.823</td>
      <td>205715.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6D0b04NJIKfEMg040WioJ</td>
      <td>Issues</td>
      <td>Julia Michaels</td>
      <td>0.706</td>
      <td>0.427</td>
      <td>8.0</td>
      <td>-6.864</td>
      <td>1.0</td>
      <td>0.0879</td>
      <td>0.413000</td>
      <td>0.000000</td>
      <td>0.0609</td>
      <td>0.4200</td>
      <td>113.804</td>
      <td>176320.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0afhq8XCExXpqazXczTSv</td>
      <td>Galway Girl</td>
      <td>Ed Sheeran</td>
      <td>0.624</td>
      <td>0.876</td>
      <td>9.0</td>
      <td>-3.374</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.073500</td>
      <td>0.000000</td>
      <td>0.3270</td>
      <td>0.7810</td>
      <td>99.943</td>
      <td>170827.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3ebXMykcMXOcLeJ9xZ17X</td>
      <td>Scared to Be Lonely</td>
      <td>Martin Garrix</td>
      <td>0.584</td>
      <td>0.540</td>
      <td>1.0</td>
      <td>-7.786</td>
      <td>0.0</td>
      <td>0.0576</td>
      <td>0.089500</td>
      <td>0.000000</td>
      <td>0.2610</td>
      <td>0.1950</td>
      <td>137.972</td>
      <td>220883.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7BKLCZ1jbUBVqRi2FVlTV</td>
      <td>Closer</td>
      <td>The Chainsmokers</td>
      <td>0.748</td>
      <td>0.524</td>
      <td>8.0</td>
      <td>-5.599</td>
      <td>1.0</td>
      <td>0.0338</td>
      <td>0.414000</td>
      <td>0.000000</td>
      <td>0.1110</td>
      <td>0.6610</td>
      <td>95.010</td>
      <td>244960.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1x5sYLZiu9r5E43kMlt9f</td>
      <td>Symphony (feat. Zara Larsson)</td>
      <td>Clean Bandit</td>
      <td>0.707</td>
      <td>0.629</td>
      <td>0.0</td>
      <td>-4.581</td>
      <td>0.0</td>
      <td>0.0563</td>
      <td>0.259000</td>
      <td>0.000016</td>
      <td>0.1380</td>
      <td>0.4570</td>
      <td>122.863</td>
      <td>212459.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5Ohxk2dO5COHF1krpoPig</td>
      <td>Sign of the Times</td>
      <td>Harry Styles</td>
      <td>0.516</td>
      <td>0.595</td>
      <td>5.0</td>
      <td>-4.630</td>
      <td>1.0</td>
      <td>0.0313</td>
      <td>0.027500</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.2220</td>
      <td>119.972</td>
      <td>340707.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>71</th>
      <td>6gBFPUFcJLzWGx4lenP6h</td>
      <td>goosebumps</td>
      <td>Travis Scott</td>
      <td>0.841</td>
      <td>0.728</td>
      <td>7.0</td>
      <td>-3.370</td>
      <td>1.0</td>
      <td>0.0484</td>
      <td>0.084700</td>
      <td>0.000000</td>
      <td>0.1490</td>
      <td>0.4300</td>
      <td>130.049</td>
      <td>243837.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>5Z3GHaZ6ec9bsiI5Benrb</td>
      <td>Young Dumb &amp; Broke</td>
      <td>Khalid</td>
      <td>0.798</td>
      <td>0.539</td>
      <td>1.0</td>
      <td>-6.351</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.199000</td>
      <td>0.000017</td>
      <td>0.1650</td>
      <td>0.3940</td>
      <td>136.949</td>
      <td>202547.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>6jA8HL9i4QGzsj6fjoxp8</td>
      <td>There for You</td>
      <td>Martin Garrix</td>
      <td>0.611</td>
      <td>0.644</td>
      <td>6.0</td>
      <td>-7.607</td>
      <td>0.0</td>
      <td>0.0553</td>
      <td>0.124000</td>
      <td>0.000000</td>
      <td>0.1240</td>
      <td>0.1300</td>
      <td>105.969</td>
      <td>221904.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>21TdkDRXuAB3k90ujRU1e</td>
      <td>Cold (feat. Future)</td>
      <td>Maroon 5</td>
      <td>0.697</td>
      <td>0.716</td>
      <td>9.0</td>
      <td>-6.288</td>
      <td>0.0</td>
      <td>0.1130</td>
      <td>0.118000</td>
      <td>0.000000</td>
      <td>0.0424</td>
      <td>0.5060</td>
      <td>99.905</td>
      <td>234308.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>75</th>
      <td>7vGuf3Y35N4wmASOKLUVV</td>
      <td>Silence</td>
      <td>Marshmello</td>
      <td>0.520</td>
      <td>0.761</td>
      <td>4.0</td>
      <td>-3.093</td>
      <td>1.0</td>
      <td>0.0853</td>
      <td>0.256000</td>
      <td>0.000005</td>
      <td>0.1700</td>
      <td>0.2860</td>
      <td>141.971</td>
      <td>180823.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1mXVgsBdtIVeCLJnSnmtd</td>
      <td>Too Good At Goodbyes</td>
      <td>Sam Smith</td>
      <td>0.698</td>
      <td>0.375</td>
      <td>5.0</td>
      <td>-8.279</td>
      <td>1.0</td>
      <td>0.0491</td>
      <td>0.652000</td>
      <td>0.000000</td>
      <td>0.1730</td>
      <td>0.5340</td>
      <td>91.920</td>
      <td>201000.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3EmmCZoqpWOTY1g2GBwJo</td>
      <td>Just Hold On</td>
      <td>Steve Aoki</td>
      <td>0.647</td>
      <td>0.932</td>
      <td>11.0</td>
      <td>-3.515</td>
      <td>1.0</td>
      <td>0.0824</td>
      <td>0.003830</td>
      <td>0.000002</td>
      <td>0.0574</td>
      <td>0.3740</td>
      <td>114.991</td>
      <td>198774.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6uFsE1JgZ20EXyU0JQZbU</td>
      <td>Look What You Made Me Do</td>
      <td>Taylor Swift</td>
      <td>0.773</td>
      <td>0.680</td>
      <td>9.0</td>
      <td>-6.378</td>
      <td>0.0</td>
      <td>0.1410</td>
      <td>0.213000</td>
      <td>0.000016</td>
      <td>0.1220</td>
      <td>0.4970</td>
      <td>128.062</td>
      <td>211859.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0CokSRCu5hZgPxcZBaEzV</td>
      <td>Glorious (feat. Skylar Grey)</td>
      <td>Macklemore</td>
      <td>0.731</td>
      <td>0.794</td>
      <td>0.0</td>
      <td>-5.126</td>
      <td>0.0</td>
      <td>0.0522</td>
      <td>0.032300</td>
      <td>0.000026</td>
      <td>0.1120</td>
      <td>0.3560</td>
      <td>139.994</td>
      <td>220454.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>6875MeXyCW0wLyT72Eetm</td>
      <td>Starving</td>
      <td>Hailee Steinfeld</td>
      <td>0.721</td>
      <td>0.626</td>
      <td>4.0</td>
      <td>-4.200</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.402000</td>
      <td>0.000000</td>
      <td>0.1020</td>
      <td>0.5580</td>
      <td>99.914</td>
      <td>181933.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>81</th>
      <td>3AEZUABDXNtecAOSC1qTf</td>
      <td>Reggaetón Lento (Bailemos)</td>
      <td>CNCO</td>
      <td>0.761</td>
      <td>0.838</td>
      <td>4.0</td>
      <td>-3.073</td>
      <td>0.0</td>
      <td>0.0502</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.1760</td>
      <td>0.7100</td>
      <td>93.974</td>
      <td>222560.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>82</th>
      <td>3E2Zh20GDCR9B1EYjfXWy</td>
      <td>Weak</td>
      <td>AJR</td>
      <td>0.673</td>
      <td>0.637</td>
      <td>5.0</td>
      <td>-4.518</td>
      <td>1.0</td>
      <td>0.0429</td>
      <td>0.137000</td>
      <td>0.000000</td>
      <td>0.1840</td>
      <td>0.6780</td>
      <td>123.980</td>
      <td>201160.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>83</th>
      <td>4pLwZjInHj3SimIyN9SnO</td>
      <td>Side To Side</td>
      <td>Ariana Grande</td>
      <td>0.648</td>
      <td>0.738</td>
      <td>6.0</td>
      <td>-5.883</td>
      <td>0.0</td>
      <td>0.2470</td>
      <td>0.040800</td>
      <td>0.000000</td>
      <td>0.2920</td>
      <td>0.6030</td>
      <td>159.145</td>
      <td>226160.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>84</th>
      <td>3QwBODjSEzelZyVjxPOHd</td>
      <td>Otra Vez (feat. J Balvin)</td>
      <td>Zion &amp; Lennox</td>
      <td>0.832</td>
      <td>0.772</td>
      <td>10.0</td>
      <td>-5.429</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.055900</td>
      <td>0.000486</td>
      <td>0.4400</td>
      <td>0.7040</td>
      <td>96.016</td>
      <td>209453.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1wjzFQodRWrPcQ0AnYnvQ</td>
      <td>I Like Me Better</td>
      <td>Lauv</td>
      <td>0.752</td>
      <td>0.505</td>
      <td>9.0</td>
      <td>-7.621</td>
      <td>1.0</td>
      <td>0.2530</td>
      <td>0.535000</td>
      <td>0.000003</td>
      <td>0.1040</td>
      <td>0.4190</td>
      <td>91.970</td>
      <td>197437.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>86</th>
      <td>04DwTuZ2VBdJCCC5TROn7</td>
      <td>In the Name of Love</td>
      <td>Martin Garrix</td>
      <td>0.490</td>
      <td>0.485</td>
      <td>4.0</td>
      <td>-6.237</td>
      <td>0.0</td>
      <td>0.0406</td>
      <td>0.059200</td>
      <td>0.000000</td>
      <td>0.3370</td>
      <td>0.1960</td>
      <td>133.889</td>
      <td>195840.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6DNtNfH8hXkqOX1sjqmI7</td>
      <td>Cold Water (feat. Justin Bieber &amp; MØ)</td>
      <td>Major Lazer</td>
      <td>0.608</td>
      <td>0.798</td>
      <td>6.0</td>
      <td>-5.092</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.073600</td>
      <td>0.000000</td>
      <td>0.1560</td>
      <td>0.5010</td>
      <td>92.943</td>
      <td>185352.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1UZOjK1BwmwWU14Erba9C</td>
      <td>Malibu</td>
      <td>Miley Cyrus</td>
      <td>0.573</td>
      <td>0.781</td>
      <td>8.0</td>
      <td>-6.406</td>
      <td>1.0</td>
      <td>0.0555</td>
      <td>0.076700</td>
      <td>0.000026</td>
      <td>0.0813</td>
      <td>0.3430</td>
      <td>139.934</td>
      <td>231907.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4b4KcovePX8Ke2cLIQTLM</td>
      <td>All Night</td>
      <td>The Vamps</td>
      <td>0.544</td>
      <td>0.809</td>
      <td>8.0</td>
      <td>-5.098</td>
      <td>1.0</td>
      <td>0.0363</td>
      <td>0.003800</td>
      <td>0.000000</td>
      <td>0.3230</td>
      <td>0.4480</td>
      <td>145.017</td>
      <td>197640.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1a5Yu5L18qNxVhXx38njO</td>
      <td>Hear Me Now</td>
      <td>Alok</td>
      <td>0.789</td>
      <td>0.442</td>
      <td>11.0</td>
      <td>-7.844</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.586000</td>
      <td>0.003660</td>
      <td>0.0927</td>
      <td>0.4500</td>
      <td>121.971</td>
      <td>192846.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4c2W3VKsOFoIg2SFaO6DY</td>
      <td>Your Song</td>
      <td>Rita Ora</td>
      <td>0.855</td>
      <td>0.624</td>
      <td>1.0</td>
      <td>-4.093</td>
      <td>1.0</td>
      <td>0.0488</td>
      <td>0.158000</td>
      <td>0.000000</td>
      <td>0.0513</td>
      <td>0.9620</td>
      <td>117.959</td>
      <td>180757.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>92</th>
      <td>22eADXu8DfOAUEDw4vU8q</td>
      <td>Ahora Dice</td>
      <td>Chris Jeday</td>
      <td>0.708</td>
      <td>0.693</td>
      <td>6.0</td>
      <td>-5.516</td>
      <td>1.0</td>
      <td>0.1380</td>
      <td>0.246000</td>
      <td>0.000000</td>
      <td>0.1290</td>
      <td>0.4270</td>
      <td>143.965</td>
      <td>271080.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>93</th>
      <td>7nZmah2llfvLDiUjm0kiy</td>
      <td>Friends (with BloodPop®)</td>
      <td>Justin Bieber</td>
      <td>0.744</td>
      <td>0.739</td>
      <td>8.0</td>
      <td>-5.350</td>
      <td>1.0</td>
      <td>0.0387</td>
      <td>0.004590</td>
      <td>0.000000</td>
      <td>0.3060</td>
      <td>0.6490</td>
      <td>104.990</td>
      <td>189467.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2fQrGHiQOvpL9UgPvtYy6</td>
      <td>Bank Account</td>
      <td>21 Savage</td>
      <td>0.884</td>
      <td>0.346</td>
      <td>8.0</td>
      <td>-8.228</td>
      <td>0.0</td>
      <td>0.3510</td>
      <td>0.015100</td>
      <td>0.000007</td>
      <td>0.0871</td>
      <td>0.3760</td>
      <td>75.016</td>
      <td>220307.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1PSBzsahR2AKwLJgx8ehB</td>
      <td>Bad Things (with Camila Cabello)</td>
      <td>Machine Gun Kelly</td>
      <td>0.675</td>
      <td>0.690</td>
      <td>2.0</td>
      <td>-4.761</td>
      <td>1.0</td>
      <td>0.1320</td>
      <td>0.210000</td>
      <td>0.000000</td>
      <td>0.2870</td>
      <td>0.2720</td>
      <td>137.817</td>
      <td>239293.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0QsvXIfqM0zZoerQfsI9l</td>
      <td>Don't Let Me Down</td>
      <td>The Chainsmokers</td>
      <td>0.542</td>
      <td>0.859</td>
      <td>11.0</td>
      <td>-5.651</td>
      <td>1.0</td>
      <td>0.1970</td>
      <td>0.160000</td>
      <td>0.004660</td>
      <td>0.1370</td>
      <td>0.4030</td>
      <td>159.797</td>
      <td>208053.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>97</th>
      <td>7mldq42yDuxiUNn08nvzH</td>
      <td>Body Like A Back Road</td>
      <td>Sam Hunt</td>
      <td>0.731</td>
      <td>0.469</td>
      <td>5.0</td>
      <td>-7.226</td>
      <td>1.0</td>
      <td>0.0326</td>
      <td>0.463000</td>
      <td>0.000001</td>
      <td>0.1030</td>
      <td>0.6310</td>
      <td>98.963</td>
      <td>165387.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>98</th>
      <td>7i2DJ88J7jQ8K7zqFX2fW</td>
      <td>Now Or Never</td>
      <td>Halsey</td>
      <td>0.658</td>
      <td>0.588</td>
      <td>6.0</td>
      <td>-4.902</td>
      <td>0.0</td>
      <td>0.0367</td>
      <td>0.105000</td>
      <td>0.000001</td>
      <td>0.1250</td>
      <td>0.4340</td>
      <td>110.075</td>
      <td>214802.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1j4kHkkpqZRBwE0A4CN4Y</td>
      <td>Dusk Till Dawn - Radio Edit</td>
      <td>ZAYN</td>
      <td>0.258</td>
      <td>0.437</td>
      <td>11.0</td>
      <td>-6.593</td>
      <td>0.0</td>
      <td>0.0390</td>
      <td>0.101000</td>
      <td>0.000001</td>
      <td>0.1060</td>
      <td>0.0967</td>
      <td>180.043</td>
      <td>239000.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 16 columns</p>
</div>




```python
print(df_transform.dtypes)
```

    id                   object
    name                 object
    artists              object
    danceability        float64
    energy              float64
    key                 float64
    loudness            float64
    mode                float64
    speechiness         float64
    acousticness        float64
    instrumentalness    float64
    liveness            float64
    valence             float64
    tempo               float64
    duration_ms         float64
    time_signature      float64
    dtype: object



```python
def func(row):
    
    return row.energy * 100

df_transform['energy_percent'] = df.apply(func,axis=1)
```


```python
print(df_transform.dtypes)
df_transform
```

    id                   object
    name                 object
    artists              object
    danceability        float64
    energy              float64
    key                 float64
    loudness            float64
    mode                float64
    speechiness         float64
    acousticness        float64
    instrumentalness    float64
    liveness            float64
    valence             float64
    tempo               float64
    duration_ms         float64
    time_signature      float64
    energy_percent      float64
    dtype: object





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>energy_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7qiZfU4dY1lWllzX7mPBI</td>
      <td>Shape of You</td>
      <td>Ed Sheeran</td>
      <td>0.825</td>
      <td>0.652</td>
      <td>1.0</td>
      <td>-3.183</td>
      <td>0.0</td>
      <td>0.0802</td>
      <td>0.581000</td>
      <td>0.000000</td>
      <td>0.0931</td>
      <td>0.9310</td>
      <td>95.977</td>
      <td>233713.0</td>
      <td>4.0</td>
      <td>65.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5CtI0qwDJkDQGwXD1H1cL</td>
      <td>Despacito - Remix</td>
      <td>Luis Fonsi</td>
      <td>0.694</td>
      <td>0.815</td>
      <td>2.0</td>
      <td>-4.328</td>
      <td>1.0</td>
      <td>0.1200</td>
      <td>0.229000</td>
      <td>0.000000</td>
      <td>0.0924</td>
      <td>0.8130</td>
      <td>88.931</td>
      <td>228827.0</td>
      <td>4.0</td>
      <td>81.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4aWmUDTfIPGksMNLV2rQP</td>
      <td>Despacito (Featuring Daddy Yankee)</td>
      <td>Luis Fonsi</td>
      <td>0.660</td>
      <td>0.786</td>
      <td>2.0</td>
      <td>-4.757</td>
      <td>1.0</td>
      <td>0.1700</td>
      <td>0.209000</td>
      <td>0.000000</td>
      <td>0.1120</td>
      <td>0.8460</td>
      <td>177.833</td>
      <td>228200.0</td>
      <td>4.0</td>
      <td>78.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6RUKPb4LETWmmr3iAEQkt</td>
      <td>Something Just Like This</td>
      <td>The Chainsmokers</td>
      <td>0.617</td>
      <td>0.635</td>
      <td>11.0</td>
      <td>-6.769</td>
      <td>0.0</td>
      <td>0.0317</td>
      <td>0.049800</td>
      <td>0.000014</td>
      <td>0.1640</td>
      <td>0.4460</td>
      <td>103.019</td>
      <td>247160.0</td>
      <td>4.0</td>
      <td>63.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3DXncPQOG4VBw3QHh3S81</td>
      <td>I'm the One</td>
      <td>DJ Khaled</td>
      <td>0.609</td>
      <td>0.668</td>
      <td>7.0</td>
      <td>-4.284</td>
      <td>1.0</td>
      <td>0.0367</td>
      <td>0.055200</td>
      <td>0.000000</td>
      <td>0.1670</td>
      <td>0.8110</td>
      <td>80.924</td>
      <td>288600.0</td>
      <td>4.0</td>
      <td>66.8</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7KXjTSCq5nL1LoYtL7XAw</td>
      <td>HUMBLE.</td>
      <td>Kendrick Lamar</td>
      <td>0.904</td>
      <td>0.611</td>
      <td>1.0</td>
      <td>-6.842</td>
      <td>0.0</td>
      <td>0.0888</td>
      <td>0.000259</td>
      <td>0.000020</td>
      <td>0.0976</td>
      <td>0.4000</td>
      <td>150.020</td>
      <td>177000.0</td>
      <td>4.0</td>
      <td>61.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3eR23VReFzcdmS7TYCrhC</td>
      <td>It Ain't Me (with Selena Gomez)</td>
      <td>Kygo</td>
      <td>0.640</td>
      <td>0.533</td>
      <td>0.0</td>
      <td>-6.596</td>
      <td>1.0</td>
      <td>0.0706</td>
      <td>0.119000</td>
      <td>0.000000</td>
      <td>0.0864</td>
      <td>0.5150</td>
      <td>99.968</td>
      <td>220781.0</td>
      <td>4.0</td>
      <td>53.3</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3B54sVLJ402zGa6Xm4YGN</td>
      <td>Unforgettable</td>
      <td>French Montana</td>
      <td>0.726</td>
      <td>0.769</td>
      <td>6.0</td>
      <td>-5.043</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.029300</td>
      <td>0.010100</td>
      <td>0.1040</td>
      <td>0.7330</td>
      <td>97.985</td>
      <td>233902.0</td>
      <td>4.0</td>
      <td>76.9</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0KKkJNfGyhkQ5aFogxQAP</td>
      <td>That's What I Like</td>
      <td>Bruno Mars</td>
      <td>0.853</td>
      <td>0.560</td>
      <td>1.0</td>
      <td>-4.961</td>
      <td>1.0</td>
      <td>0.0406</td>
      <td>0.013000</td>
      <td>0.000000</td>
      <td>0.0944</td>
      <td>0.8600</td>
      <td>134.066</td>
      <td>206693.0</td>
      <td>4.0</td>
      <td>56.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3NdDpSvN911VPGivFlV5d</td>
      <td>I Don’t Wanna Live Forever (Fifty Shades Darke...</td>
      <td>ZAYN</td>
      <td>0.735</td>
      <td>0.451</td>
      <td>0.0</td>
      <td>-8.374</td>
      <td>1.0</td>
      <td>0.0585</td>
      <td>0.063100</td>
      <td>0.000013</td>
      <td>0.3250</td>
      <td>0.0862</td>
      <td>117.973</td>
      <td>245200.0</td>
      <td>4.0</td>
      <td>45.1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7GX5flRQZVHRAGd6B4TmD</td>
      <td>XO TOUR Llif3</td>
      <td>Lil Uzi Vert</td>
      <td>0.732</td>
      <td>0.750</td>
      <td>11.0</td>
      <td>-6.366</td>
      <td>0.0</td>
      <td>0.2310</td>
      <td>0.002640</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.4010</td>
      <td>155.096</td>
      <td>182707.0</td>
      <td>4.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>72jbDTw1piOOj770jWNea</td>
      <td>Paris</td>
      <td>The Chainsmokers</td>
      <td>0.653</td>
      <td>0.658</td>
      <td>2.0</td>
      <td>-6.428</td>
      <td>1.0</td>
      <td>0.0304</td>
      <td>0.021500</td>
      <td>0.000002</td>
      <td>0.0939</td>
      <td>0.2190</td>
      <td>99.990</td>
      <td>221507.0</td>
      <td>4.0</td>
      <td>65.8</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0dA2Mk56wEzDgegdC6R17</td>
      <td>Stay (with Alessia Cara)</td>
      <td>Zedd</td>
      <td>0.679</td>
      <td>0.634</td>
      <td>5.0</td>
      <td>-5.024</td>
      <td>0.0</td>
      <td>0.0654</td>
      <td>0.232000</td>
      <td>0.000000</td>
      <td>0.1150</td>
      <td>0.4980</td>
      <td>102.013</td>
      <td>210091.0</td>
      <td>4.0</td>
      <td>63.4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4iLqG9SeJSnt0cSPICSjx</td>
      <td>Attention</td>
      <td>Charlie Puth</td>
      <td>0.774</td>
      <td>0.626</td>
      <td>3.0</td>
      <td>-4.432</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.096900</td>
      <td>0.000031</td>
      <td>0.0848</td>
      <td>0.7770</td>
      <td>100.041</td>
      <td>211475.0</td>
      <td>4.0</td>
      <td>62.6</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0VgkVdmE4gld66l8iyGjg</td>
      <td>Mask Off</td>
      <td>Future</td>
      <td>0.833</td>
      <td>0.434</td>
      <td>2.0</td>
      <td>-8.795</td>
      <td>1.0</td>
      <td>0.4310</td>
      <td>0.010200</td>
      <td>0.021900</td>
      <td>0.1650</td>
      <td>0.2810</td>
      <td>150.062</td>
      <td>204600.0</td>
      <td>4.0</td>
      <td>43.4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3a1lNhkSLSkpJE4MSHpDu</td>
      <td>Congratulations</td>
      <td>Post Malone</td>
      <td>0.627</td>
      <td>0.812</td>
      <td>6.0</td>
      <td>-4.215</td>
      <td>1.0</td>
      <td>0.0358</td>
      <td>0.198000</td>
      <td>0.000000</td>
      <td>0.2120</td>
      <td>0.5040</td>
      <td>123.071</td>
      <td>220293.0</td>
      <td>4.0</td>
      <td>81.2</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6kex4EBAj0WHXDKZMEJaa</td>
      <td>Swalla (feat. Nicki Minaj &amp; Ty Dolla $ign)</td>
      <td>Jason Derulo</td>
      <td>0.696</td>
      <td>0.817</td>
      <td>1.0</td>
      <td>-3.862</td>
      <td>1.0</td>
      <td>0.1090</td>
      <td>0.075000</td>
      <td>0.000000</td>
      <td>0.1870</td>
      <td>0.7820</td>
      <td>98.064</td>
      <td>216409.0</td>
      <td>4.0</td>
      <td>81.7</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6PCUP3dWmTjcTtXY02oFd</td>
      <td>Castle on the Hill</td>
      <td>Ed Sheeran</td>
      <td>0.461</td>
      <td>0.834</td>
      <td>2.0</td>
      <td>-4.868</td>
      <td>1.0</td>
      <td>0.0989</td>
      <td>0.023200</td>
      <td>0.000011</td>
      <td>0.1400</td>
      <td>0.4710</td>
      <td>135.007</td>
      <td>261154.0</td>
      <td>4.0</td>
      <td>83.4</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5knuzwU65gJK7IF5yJsua</td>
      <td>Rockabye (feat. Sean Paul &amp; Anne-Marie)</td>
      <td>Clean Bandit</td>
      <td>0.720</td>
      <td>0.763</td>
      <td>9.0</td>
      <td>-4.068</td>
      <td>0.0</td>
      <td>0.0523</td>
      <td>0.406000</td>
      <td>0.000000</td>
      <td>0.1800</td>
      <td>0.7420</td>
      <td>101.965</td>
      <td>251088.0</td>
      <td>4.0</td>
      <td>76.3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0CcQNd8CINkwQfe1RDtGV</td>
      <td>Believer</td>
      <td>Imagine Dragons</td>
      <td>0.779</td>
      <td>0.787</td>
      <td>10.0</td>
      <td>-4.305</td>
      <td>0.0</td>
      <td>0.1080</td>
      <td>0.052400</td>
      <td>0.000000</td>
      <td>0.1400</td>
      <td>0.7080</td>
      <td>124.982</td>
      <td>204347.0</td>
      <td>4.0</td>
      <td>78.7</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2rb5MvYT7ZIxbKW5hfcHx</td>
      <td>Mi Gente</td>
      <td>J Balvin</td>
      <td>0.543</td>
      <td>0.677</td>
      <td>11.0</td>
      <td>-4.915</td>
      <td>0.0</td>
      <td>0.0993</td>
      <td>0.014800</td>
      <td>0.000006</td>
      <td>0.1300</td>
      <td>0.2940</td>
      <td>103.809</td>
      <td>189440.0</td>
      <td>4.0</td>
      <td>67.7</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0tKcYR2II1VCQWT79i5Nr</td>
      <td>Thunder</td>
      <td>Imagine Dragons</td>
      <td>0.600</td>
      <td>0.810</td>
      <td>0.0</td>
      <td>-4.749</td>
      <td>1.0</td>
      <td>0.0479</td>
      <td>0.006830</td>
      <td>0.210000</td>
      <td>0.1550</td>
      <td>0.2980</td>
      <td>167.880</td>
      <td>187147.0</td>
      <td>4.0</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5uCax9HTNlzGybIStD3vD</td>
      <td>Say You Won't Let Go</td>
      <td>James Arthur</td>
      <td>0.358</td>
      <td>0.557</td>
      <td>10.0</td>
      <td>-7.398</td>
      <td>1.0</td>
      <td>0.0590</td>
      <td>0.695000</td>
      <td>0.000000</td>
      <td>0.0902</td>
      <td>0.4940</td>
      <td>85.043</td>
      <td>211467.0</td>
      <td>4.0</td>
      <td>55.7</td>
    </tr>
    <tr>
      <th>23</th>
      <td>79cuOz3SPQTuFrp8WgftA</td>
      <td>There's Nothing Holdin' Me Back</td>
      <td>Shawn Mendes</td>
      <td>0.857</td>
      <td>0.800</td>
      <td>2.0</td>
      <td>-4.035</td>
      <td>1.0</td>
      <td>0.0583</td>
      <td>0.381000</td>
      <td>0.000000</td>
      <td>0.0913</td>
      <td>0.9660</td>
      <td>121.996</td>
      <td>199440.0</td>
      <td>4.0</td>
      <td>80.0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6De0lHrwBfPfrhorm9q1X</td>
      <td>Me Rehúso</td>
      <td>Danny Ocean</td>
      <td>0.744</td>
      <td>0.804</td>
      <td>1.0</td>
      <td>-6.327</td>
      <td>1.0</td>
      <td>0.0677</td>
      <td>0.023100</td>
      <td>0.000000</td>
      <td>0.0494</td>
      <td>0.4260</td>
      <td>104.823</td>
      <td>205715.0</td>
      <td>4.0</td>
      <td>80.4</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6D0b04NJIKfEMg040WioJ</td>
      <td>Issues</td>
      <td>Julia Michaels</td>
      <td>0.706</td>
      <td>0.427</td>
      <td>8.0</td>
      <td>-6.864</td>
      <td>1.0</td>
      <td>0.0879</td>
      <td>0.413000</td>
      <td>0.000000</td>
      <td>0.0609</td>
      <td>0.4200</td>
      <td>113.804</td>
      <td>176320.0</td>
      <td>4.0</td>
      <td>42.7</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0afhq8XCExXpqazXczTSv</td>
      <td>Galway Girl</td>
      <td>Ed Sheeran</td>
      <td>0.624</td>
      <td>0.876</td>
      <td>9.0</td>
      <td>-3.374</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.073500</td>
      <td>0.000000</td>
      <td>0.3270</td>
      <td>0.7810</td>
      <td>99.943</td>
      <td>170827.0</td>
      <td>4.0</td>
      <td>87.6</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3ebXMykcMXOcLeJ9xZ17X</td>
      <td>Scared to Be Lonely</td>
      <td>Martin Garrix</td>
      <td>0.584</td>
      <td>0.540</td>
      <td>1.0</td>
      <td>-7.786</td>
      <td>0.0</td>
      <td>0.0576</td>
      <td>0.089500</td>
      <td>0.000000</td>
      <td>0.2610</td>
      <td>0.1950</td>
      <td>137.972</td>
      <td>220883.0</td>
      <td>4.0</td>
      <td>54.0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7BKLCZ1jbUBVqRi2FVlTV</td>
      <td>Closer</td>
      <td>The Chainsmokers</td>
      <td>0.748</td>
      <td>0.524</td>
      <td>8.0</td>
      <td>-5.599</td>
      <td>1.0</td>
      <td>0.0338</td>
      <td>0.414000</td>
      <td>0.000000</td>
      <td>0.1110</td>
      <td>0.6610</td>
      <td>95.010</td>
      <td>244960.0</td>
      <td>4.0</td>
      <td>52.4</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1x5sYLZiu9r5E43kMlt9f</td>
      <td>Symphony (feat. Zara Larsson)</td>
      <td>Clean Bandit</td>
      <td>0.707</td>
      <td>0.629</td>
      <td>0.0</td>
      <td>-4.581</td>
      <td>0.0</td>
      <td>0.0563</td>
      <td>0.259000</td>
      <td>0.000016</td>
      <td>0.1380</td>
      <td>0.4570</td>
      <td>122.863</td>
      <td>212459.0</td>
      <td>4.0</td>
      <td>62.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5Ohxk2dO5COHF1krpoPig</td>
      <td>Sign of the Times</td>
      <td>Harry Styles</td>
      <td>0.516</td>
      <td>0.595</td>
      <td>5.0</td>
      <td>-4.630</td>
      <td>1.0</td>
      <td>0.0313</td>
      <td>0.027500</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.2220</td>
      <td>119.972</td>
      <td>340707.0</td>
      <td>4.0</td>
      <td>59.5</td>
    </tr>
    <tr>
      <th>71</th>
      <td>6gBFPUFcJLzWGx4lenP6h</td>
      <td>goosebumps</td>
      <td>Travis Scott</td>
      <td>0.841</td>
      <td>0.728</td>
      <td>7.0</td>
      <td>-3.370</td>
      <td>1.0</td>
      <td>0.0484</td>
      <td>0.084700</td>
      <td>0.000000</td>
      <td>0.1490</td>
      <td>0.4300</td>
      <td>130.049</td>
      <td>243837.0</td>
      <td>4.0</td>
      <td>72.8</td>
    </tr>
    <tr>
      <th>72</th>
      <td>5Z3GHaZ6ec9bsiI5Benrb</td>
      <td>Young Dumb &amp; Broke</td>
      <td>Khalid</td>
      <td>0.798</td>
      <td>0.539</td>
      <td>1.0</td>
      <td>-6.351</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.199000</td>
      <td>0.000017</td>
      <td>0.1650</td>
      <td>0.3940</td>
      <td>136.949</td>
      <td>202547.0</td>
      <td>4.0</td>
      <td>53.9</td>
    </tr>
    <tr>
      <th>73</th>
      <td>6jA8HL9i4QGzsj6fjoxp8</td>
      <td>There for You</td>
      <td>Martin Garrix</td>
      <td>0.611</td>
      <td>0.644</td>
      <td>6.0</td>
      <td>-7.607</td>
      <td>0.0</td>
      <td>0.0553</td>
      <td>0.124000</td>
      <td>0.000000</td>
      <td>0.1240</td>
      <td>0.1300</td>
      <td>105.969</td>
      <td>221904.0</td>
      <td>4.0</td>
      <td>64.4</td>
    </tr>
    <tr>
      <th>74</th>
      <td>21TdkDRXuAB3k90ujRU1e</td>
      <td>Cold (feat. Future)</td>
      <td>Maroon 5</td>
      <td>0.697</td>
      <td>0.716</td>
      <td>9.0</td>
      <td>-6.288</td>
      <td>0.0</td>
      <td>0.1130</td>
      <td>0.118000</td>
      <td>0.000000</td>
      <td>0.0424</td>
      <td>0.5060</td>
      <td>99.905</td>
      <td>234308.0</td>
      <td>4.0</td>
      <td>71.6</td>
    </tr>
    <tr>
      <th>75</th>
      <td>7vGuf3Y35N4wmASOKLUVV</td>
      <td>Silence</td>
      <td>Marshmello</td>
      <td>0.520</td>
      <td>0.761</td>
      <td>4.0</td>
      <td>-3.093</td>
      <td>1.0</td>
      <td>0.0853</td>
      <td>0.256000</td>
      <td>0.000005</td>
      <td>0.1700</td>
      <td>0.2860</td>
      <td>141.971</td>
      <td>180823.0</td>
      <td>4.0</td>
      <td>76.1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1mXVgsBdtIVeCLJnSnmtd</td>
      <td>Too Good At Goodbyes</td>
      <td>Sam Smith</td>
      <td>0.698</td>
      <td>0.375</td>
      <td>5.0</td>
      <td>-8.279</td>
      <td>1.0</td>
      <td>0.0491</td>
      <td>0.652000</td>
      <td>0.000000</td>
      <td>0.1730</td>
      <td>0.5340</td>
      <td>91.920</td>
      <td>201000.0</td>
      <td>4.0</td>
      <td>37.5</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3EmmCZoqpWOTY1g2GBwJo</td>
      <td>Just Hold On</td>
      <td>Steve Aoki</td>
      <td>0.647</td>
      <td>0.932</td>
      <td>11.0</td>
      <td>-3.515</td>
      <td>1.0</td>
      <td>0.0824</td>
      <td>0.003830</td>
      <td>0.000002</td>
      <td>0.0574</td>
      <td>0.3740</td>
      <td>114.991</td>
      <td>198774.0</td>
      <td>4.0</td>
      <td>93.2</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6uFsE1JgZ20EXyU0JQZbU</td>
      <td>Look What You Made Me Do</td>
      <td>Taylor Swift</td>
      <td>0.773</td>
      <td>0.680</td>
      <td>9.0</td>
      <td>-6.378</td>
      <td>0.0</td>
      <td>0.1410</td>
      <td>0.213000</td>
      <td>0.000016</td>
      <td>0.1220</td>
      <td>0.4970</td>
      <td>128.062</td>
      <td>211859.0</td>
      <td>4.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0CokSRCu5hZgPxcZBaEzV</td>
      <td>Glorious (feat. Skylar Grey)</td>
      <td>Macklemore</td>
      <td>0.731</td>
      <td>0.794</td>
      <td>0.0</td>
      <td>-5.126</td>
      <td>0.0</td>
      <td>0.0522</td>
      <td>0.032300</td>
      <td>0.000026</td>
      <td>0.1120</td>
      <td>0.3560</td>
      <td>139.994</td>
      <td>220454.0</td>
      <td>4.0</td>
      <td>79.4</td>
    </tr>
    <tr>
      <th>80</th>
      <td>6875MeXyCW0wLyT72Eetm</td>
      <td>Starving</td>
      <td>Hailee Steinfeld</td>
      <td>0.721</td>
      <td>0.626</td>
      <td>4.0</td>
      <td>-4.200</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.402000</td>
      <td>0.000000</td>
      <td>0.1020</td>
      <td>0.5580</td>
      <td>99.914</td>
      <td>181933.0</td>
      <td>4.0</td>
      <td>62.6</td>
    </tr>
    <tr>
      <th>81</th>
      <td>3AEZUABDXNtecAOSC1qTf</td>
      <td>Reggaetón Lento (Bailemos)</td>
      <td>CNCO</td>
      <td>0.761</td>
      <td>0.838</td>
      <td>4.0</td>
      <td>-3.073</td>
      <td>0.0</td>
      <td>0.0502</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.1760</td>
      <td>0.7100</td>
      <td>93.974</td>
      <td>222560.0</td>
      <td>4.0</td>
      <td>83.8</td>
    </tr>
    <tr>
      <th>82</th>
      <td>3E2Zh20GDCR9B1EYjfXWy</td>
      <td>Weak</td>
      <td>AJR</td>
      <td>0.673</td>
      <td>0.637</td>
      <td>5.0</td>
      <td>-4.518</td>
      <td>1.0</td>
      <td>0.0429</td>
      <td>0.137000</td>
      <td>0.000000</td>
      <td>0.1840</td>
      <td>0.6780</td>
      <td>123.980</td>
      <td>201160.0</td>
      <td>4.0</td>
      <td>63.7</td>
    </tr>
    <tr>
      <th>83</th>
      <td>4pLwZjInHj3SimIyN9SnO</td>
      <td>Side To Side</td>
      <td>Ariana Grande</td>
      <td>0.648</td>
      <td>0.738</td>
      <td>6.0</td>
      <td>-5.883</td>
      <td>0.0</td>
      <td>0.2470</td>
      <td>0.040800</td>
      <td>0.000000</td>
      <td>0.2920</td>
      <td>0.6030</td>
      <td>159.145</td>
      <td>226160.0</td>
      <td>4.0</td>
      <td>73.8</td>
    </tr>
    <tr>
      <th>84</th>
      <td>3QwBODjSEzelZyVjxPOHd</td>
      <td>Otra Vez (feat. J Balvin)</td>
      <td>Zion &amp; Lennox</td>
      <td>0.832</td>
      <td>0.772</td>
      <td>10.0</td>
      <td>-5.429</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.055900</td>
      <td>0.000486</td>
      <td>0.4400</td>
      <td>0.7040</td>
      <td>96.016</td>
      <td>209453.0</td>
      <td>4.0</td>
      <td>77.2</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1wjzFQodRWrPcQ0AnYnvQ</td>
      <td>I Like Me Better</td>
      <td>Lauv</td>
      <td>0.752</td>
      <td>0.505</td>
      <td>9.0</td>
      <td>-7.621</td>
      <td>1.0</td>
      <td>0.2530</td>
      <td>0.535000</td>
      <td>0.000003</td>
      <td>0.1040</td>
      <td>0.4190</td>
      <td>91.970</td>
      <td>197437.0</td>
      <td>4.0</td>
      <td>50.5</td>
    </tr>
    <tr>
      <th>86</th>
      <td>04DwTuZ2VBdJCCC5TROn7</td>
      <td>In the Name of Love</td>
      <td>Martin Garrix</td>
      <td>0.490</td>
      <td>0.485</td>
      <td>4.0</td>
      <td>-6.237</td>
      <td>0.0</td>
      <td>0.0406</td>
      <td>0.059200</td>
      <td>0.000000</td>
      <td>0.3370</td>
      <td>0.1960</td>
      <td>133.889</td>
      <td>195840.0</td>
      <td>4.0</td>
      <td>48.5</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6DNtNfH8hXkqOX1sjqmI7</td>
      <td>Cold Water (feat. Justin Bieber &amp; MØ)</td>
      <td>Major Lazer</td>
      <td>0.608</td>
      <td>0.798</td>
      <td>6.0</td>
      <td>-5.092</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.073600</td>
      <td>0.000000</td>
      <td>0.1560</td>
      <td>0.5010</td>
      <td>92.943</td>
      <td>185352.0</td>
      <td>4.0</td>
      <td>79.8</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1UZOjK1BwmwWU14Erba9C</td>
      <td>Malibu</td>
      <td>Miley Cyrus</td>
      <td>0.573</td>
      <td>0.781</td>
      <td>8.0</td>
      <td>-6.406</td>
      <td>1.0</td>
      <td>0.0555</td>
      <td>0.076700</td>
      <td>0.000026</td>
      <td>0.0813</td>
      <td>0.3430</td>
      <td>139.934</td>
      <td>231907.0</td>
      <td>4.0</td>
      <td>78.1</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4b4KcovePX8Ke2cLIQTLM</td>
      <td>All Night</td>
      <td>The Vamps</td>
      <td>0.544</td>
      <td>0.809</td>
      <td>8.0</td>
      <td>-5.098</td>
      <td>1.0</td>
      <td>0.0363</td>
      <td>0.003800</td>
      <td>0.000000</td>
      <td>0.3230</td>
      <td>0.4480</td>
      <td>145.017</td>
      <td>197640.0</td>
      <td>4.0</td>
      <td>80.9</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1a5Yu5L18qNxVhXx38njO</td>
      <td>Hear Me Now</td>
      <td>Alok</td>
      <td>0.789</td>
      <td>0.442</td>
      <td>11.0</td>
      <td>-7.844</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.586000</td>
      <td>0.003660</td>
      <td>0.0927</td>
      <td>0.4500</td>
      <td>121.971</td>
      <td>192846.0</td>
      <td>4.0</td>
      <td>44.2</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4c2W3VKsOFoIg2SFaO6DY</td>
      <td>Your Song</td>
      <td>Rita Ora</td>
      <td>0.855</td>
      <td>0.624</td>
      <td>1.0</td>
      <td>-4.093</td>
      <td>1.0</td>
      <td>0.0488</td>
      <td>0.158000</td>
      <td>0.000000</td>
      <td>0.0513</td>
      <td>0.9620</td>
      <td>117.959</td>
      <td>180757.0</td>
      <td>4.0</td>
      <td>62.4</td>
    </tr>
    <tr>
      <th>92</th>
      <td>22eADXu8DfOAUEDw4vU8q</td>
      <td>Ahora Dice</td>
      <td>Chris Jeday</td>
      <td>0.708</td>
      <td>0.693</td>
      <td>6.0</td>
      <td>-5.516</td>
      <td>1.0</td>
      <td>0.1380</td>
      <td>0.246000</td>
      <td>0.000000</td>
      <td>0.1290</td>
      <td>0.4270</td>
      <td>143.965</td>
      <td>271080.0</td>
      <td>4.0</td>
      <td>69.3</td>
    </tr>
    <tr>
      <th>93</th>
      <td>7nZmah2llfvLDiUjm0kiy</td>
      <td>Friends (with BloodPop®)</td>
      <td>Justin Bieber</td>
      <td>0.744</td>
      <td>0.739</td>
      <td>8.0</td>
      <td>-5.350</td>
      <td>1.0</td>
      <td>0.0387</td>
      <td>0.004590</td>
      <td>0.000000</td>
      <td>0.3060</td>
      <td>0.6490</td>
      <td>104.990</td>
      <td>189467.0</td>
      <td>4.0</td>
      <td>73.9</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2fQrGHiQOvpL9UgPvtYy6</td>
      <td>Bank Account</td>
      <td>21 Savage</td>
      <td>0.884</td>
      <td>0.346</td>
      <td>8.0</td>
      <td>-8.228</td>
      <td>0.0</td>
      <td>0.3510</td>
      <td>0.015100</td>
      <td>0.000007</td>
      <td>0.0871</td>
      <td>0.3760</td>
      <td>75.016</td>
      <td>220307.0</td>
      <td>4.0</td>
      <td>34.6</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1PSBzsahR2AKwLJgx8ehB</td>
      <td>Bad Things (with Camila Cabello)</td>
      <td>Machine Gun Kelly</td>
      <td>0.675</td>
      <td>0.690</td>
      <td>2.0</td>
      <td>-4.761</td>
      <td>1.0</td>
      <td>0.1320</td>
      <td>0.210000</td>
      <td>0.000000</td>
      <td>0.2870</td>
      <td>0.2720</td>
      <td>137.817</td>
      <td>239293.0</td>
      <td>4.0</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0QsvXIfqM0zZoerQfsI9l</td>
      <td>Don't Let Me Down</td>
      <td>The Chainsmokers</td>
      <td>0.542</td>
      <td>0.859</td>
      <td>11.0</td>
      <td>-5.651</td>
      <td>1.0</td>
      <td>0.1970</td>
      <td>0.160000</td>
      <td>0.004660</td>
      <td>0.1370</td>
      <td>0.4030</td>
      <td>159.797</td>
      <td>208053.0</td>
      <td>4.0</td>
      <td>85.9</td>
    </tr>
    <tr>
      <th>97</th>
      <td>7mldq42yDuxiUNn08nvzH</td>
      <td>Body Like A Back Road</td>
      <td>Sam Hunt</td>
      <td>0.731</td>
      <td>0.469</td>
      <td>5.0</td>
      <td>-7.226</td>
      <td>1.0</td>
      <td>0.0326</td>
      <td>0.463000</td>
      <td>0.000001</td>
      <td>0.1030</td>
      <td>0.6310</td>
      <td>98.963</td>
      <td>165387.0</td>
      <td>4.0</td>
      <td>46.9</td>
    </tr>
    <tr>
      <th>98</th>
      <td>7i2DJ88J7jQ8K7zqFX2fW</td>
      <td>Now Or Never</td>
      <td>Halsey</td>
      <td>0.658</td>
      <td>0.588</td>
      <td>6.0</td>
      <td>-4.902</td>
      <td>0.0</td>
      <td>0.0367</td>
      <td>0.105000</td>
      <td>0.000001</td>
      <td>0.1250</td>
      <td>0.4340</td>
      <td>110.075</td>
      <td>214802.0</td>
      <td>4.0</td>
      <td>58.8</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1j4kHkkpqZRBwE0A4CN4Y</td>
      <td>Dusk Till Dawn - Radio Edit</td>
      <td>ZAYN</td>
      <td>0.258</td>
      <td>0.437</td>
      <td>11.0</td>
      <td>-6.593</td>
      <td>0.0</td>
      <td>0.0390</td>
      <td>0.101000</td>
      <td>0.000001</td>
      <td>0.1060</td>
      <td>0.0967</td>
      <td>180.043</td>
      <td>239000.0</td>
      <td>4.0</td>
      <td>43.7</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 17 columns</p>
</div>




```python
def func(row):
    if row['key'] == 0.0:
        return 'C'
    elif row['key'] == 1.0:
        return 'C#/Db' 
    elif row['key'] == 2.0:
        return 'D'
    elif row['key'] == 3.0:
        return 'Eb'
    elif row['key'] == 4.0:
        return 'E'
    elif row['key'] == 5.0:
        return 'F'
    elif row['key'] == 6.0:
        return 'F#/Gb'
    elif row['key'] == 7.0:
        return 'G'
    elif row['key'] == 8.0:
        return 'Ab'
    elif row['key'] == 9.0:
        return 'A'
    elif row['key'] == 10.0:
        return 'Bb'
    elif row['key'] == 11:
        return 'B'   
    else:
        return 'N/A'

df_transform['key_change'] = df.apply(func,axis=1)
df_transform
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>name</th>
      <th>artists</th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>energy_percent</th>
      <th>key_change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7qiZfU4dY1lWllzX7mPBI</td>
      <td>Shape of You</td>
      <td>Ed Sheeran</td>
      <td>0.825</td>
      <td>0.652</td>
      <td>1.0</td>
      <td>-3.183</td>
      <td>0.0</td>
      <td>0.0802</td>
      <td>0.581000</td>
      <td>0.000000</td>
      <td>0.0931</td>
      <td>0.9310</td>
      <td>95.977</td>
      <td>233713.0</td>
      <td>4.0</td>
      <td>65.2</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5CtI0qwDJkDQGwXD1H1cL</td>
      <td>Despacito - Remix</td>
      <td>Luis Fonsi</td>
      <td>0.694</td>
      <td>0.815</td>
      <td>2.0</td>
      <td>-4.328</td>
      <td>1.0</td>
      <td>0.1200</td>
      <td>0.229000</td>
      <td>0.000000</td>
      <td>0.0924</td>
      <td>0.8130</td>
      <td>88.931</td>
      <td>228827.0</td>
      <td>4.0</td>
      <td>81.5</td>
      <td>D</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4aWmUDTfIPGksMNLV2rQP</td>
      <td>Despacito (Featuring Daddy Yankee)</td>
      <td>Luis Fonsi</td>
      <td>0.660</td>
      <td>0.786</td>
      <td>2.0</td>
      <td>-4.757</td>
      <td>1.0</td>
      <td>0.1700</td>
      <td>0.209000</td>
      <td>0.000000</td>
      <td>0.1120</td>
      <td>0.8460</td>
      <td>177.833</td>
      <td>228200.0</td>
      <td>4.0</td>
      <td>78.6</td>
      <td>D</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6RUKPb4LETWmmr3iAEQkt</td>
      <td>Something Just Like This</td>
      <td>The Chainsmokers</td>
      <td>0.617</td>
      <td>0.635</td>
      <td>11.0</td>
      <td>-6.769</td>
      <td>0.0</td>
      <td>0.0317</td>
      <td>0.049800</td>
      <td>0.000014</td>
      <td>0.1640</td>
      <td>0.4460</td>
      <td>103.019</td>
      <td>247160.0</td>
      <td>4.0</td>
      <td>63.5</td>
      <td>B</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3DXncPQOG4VBw3QHh3S81</td>
      <td>I'm the One</td>
      <td>DJ Khaled</td>
      <td>0.609</td>
      <td>0.668</td>
      <td>7.0</td>
      <td>-4.284</td>
      <td>1.0</td>
      <td>0.0367</td>
      <td>0.055200</td>
      <td>0.000000</td>
      <td>0.1670</td>
      <td>0.8110</td>
      <td>80.924</td>
      <td>288600.0</td>
      <td>4.0</td>
      <td>66.8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7KXjTSCq5nL1LoYtL7XAw</td>
      <td>HUMBLE.</td>
      <td>Kendrick Lamar</td>
      <td>0.904</td>
      <td>0.611</td>
      <td>1.0</td>
      <td>-6.842</td>
      <td>0.0</td>
      <td>0.0888</td>
      <td>0.000259</td>
      <td>0.000020</td>
      <td>0.0976</td>
      <td>0.4000</td>
      <td>150.020</td>
      <td>177000.0</td>
      <td>4.0</td>
      <td>61.1</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3eR23VReFzcdmS7TYCrhC</td>
      <td>It Ain't Me (with Selena Gomez)</td>
      <td>Kygo</td>
      <td>0.640</td>
      <td>0.533</td>
      <td>0.0</td>
      <td>-6.596</td>
      <td>1.0</td>
      <td>0.0706</td>
      <td>0.119000</td>
      <td>0.000000</td>
      <td>0.0864</td>
      <td>0.5150</td>
      <td>99.968</td>
      <td>220781.0</td>
      <td>4.0</td>
      <td>53.3</td>
      <td>C</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3B54sVLJ402zGa6Xm4YGN</td>
      <td>Unforgettable</td>
      <td>French Montana</td>
      <td>0.726</td>
      <td>0.769</td>
      <td>6.0</td>
      <td>-5.043</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.029300</td>
      <td>0.010100</td>
      <td>0.1040</td>
      <td>0.7330</td>
      <td>97.985</td>
      <td>233902.0</td>
      <td>4.0</td>
      <td>76.9</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0KKkJNfGyhkQ5aFogxQAP</td>
      <td>That's What I Like</td>
      <td>Bruno Mars</td>
      <td>0.853</td>
      <td>0.560</td>
      <td>1.0</td>
      <td>-4.961</td>
      <td>1.0</td>
      <td>0.0406</td>
      <td>0.013000</td>
      <td>0.000000</td>
      <td>0.0944</td>
      <td>0.8600</td>
      <td>134.066</td>
      <td>206693.0</td>
      <td>4.0</td>
      <td>56.0</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3NdDpSvN911VPGivFlV5d</td>
      <td>I Don’t Wanna Live Forever (Fifty Shades Darke...</td>
      <td>ZAYN</td>
      <td>0.735</td>
      <td>0.451</td>
      <td>0.0</td>
      <td>-8.374</td>
      <td>1.0</td>
      <td>0.0585</td>
      <td>0.063100</td>
      <td>0.000013</td>
      <td>0.3250</td>
      <td>0.0862</td>
      <td>117.973</td>
      <td>245200.0</td>
      <td>4.0</td>
      <td>45.1</td>
      <td>C</td>
    </tr>
    <tr>
      <th>10</th>
      <td>7GX5flRQZVHRAGd6B4TmD</td>
      <td>XO TOUR Llif3</td>
      <td>Lil Uzi Vert</td>
      <td>0.732</td>
      <td>0.750</td>
      <td>11.0</td>
      <td>-6.366</td>
      <td>0.0</td>
      <td>0.2310</td>
      <td>0.002640</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.4010</td>
      <td>155.096</td>
      <td>182707.0</td>
      <td>4.0</td>
      <td>75.0</td>
      <td>B</td>
    </tr>
    <tr>
      <th>11</th>
      <td>72jbDTw1piOOj770jWNea</td>
      <td>Paris</td>
      <td>The Chainsmokers</td>
      <td>0.653</td>
      <td>0.658</td>
      <td>2.0</td>
      <td>-6.428</td>
      <td>1.0</td>
      <td>0.0304</td>
      <td>0.021500</td>
      <td>0.000002</td>
      <td>0.0939</td>
      <td>0.2190</td>
      <td>99.990</td>
      <td>221507.0</td>
      <td>4.0</td>
      <td>65.8</td>
      <td>D</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0dA2Mk56wEzDgegdC6R17</td>
      <td>Stay (with Alessia Cara)</td>
      <td>Zedd</td>
      <td>0.679</td>
      <td>0.634</td>
      <td>5.0</td>
      <td>-5.024</td>
      <td>0.0</td>
      <td>0.0654</td>
      <td>0.232000</td>
      <td>0.000000</td>
      <td>0.1150</td>
      <td>0.4980</td>
      <td>102.013</td>
      <td>210091.0</td>
      <td>4.0</td>
      <td>63.4</td>
      <td>F</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4iLqG9SeJSnt0cSPICSjx</td>
      <td>Attention</td>
      <td>Charlie Puth</td>
      <td>0.774</td>
      <td>0.626</td>
      <td>3.0</td>
      <td>-4.432</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.096900</td>
      <td>0.000031</td>
      <td>0.0848</td>
      <td>0.7770</td>
      <td>100.041</td>
      <td>211475.0</td>
      <td>4.0</td>
      <td>62.6</td>
      <td>Eb</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0VgkVdmE4gld66l8iyGjg</td>
      <td>Mask Off</td>
      <td>Future</td>
      <td>0.833</td>
      <td>0.434</td>
      <td>2.0</td>
      <td>-8.795</td>
      <td>1.0</td>
      <td>0.4310</td>
      <td>0.010200</td>
      <td>0.021900</td>
      <td>0.1650</td>
      <td>0.2810</td>
      <td>150.062</td>
      <td>204600.0</td>
      <td>4.0</td>
      <td>43.4</td>
      <td>D</td>
    </tr>
    <tr>
      <th>15</th>
      <td>3a1lNhkSLSkpJE4MSHpDu</td>
      <td>Congratulations</td>
      <td>Post Malone</td>
      <td>0.627</td>
      <td>0.812</td>
      <td>6.0</td>
      <td>-4.215</td>
      <td>1.0</td>
      <td>0.0358</td>
      <td>0.198000</td>
      <td>0.000000</td>
      <td>0.2120</td>
      <td>0.5040</td>
      <td>123.071</td>
      <td>220293.0</td>
      <td>4.0</td>
      <td>81.2</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>16</th>
      <td>6kex4EBAj0WHXDKZMEJaa</td>
      <td>Swalla (feat. Nicki Minaj &amp; Ty Dolla $ign)</td>
      <td>Jason Derulo</td>
      <td>0.696</td>
      <td>0.817</td>
      <td>1.0</td>
      <td>-3.862</td>
      <td>1.0</td>
      <td>0.1090</td>
      <td>0.075000</td>
      <td>0.000000</td>
      <td>0.1870</td>
      <td>0.7820</td>
      <td>98.064</td>
      <td>216409.0</td>
      <td>4.0</td>
      <td>81.7</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>17</th>
      <td>6PCUP3dWmTjcTtXY02oFd</td>
      <td>Castle on the Hill</td>
      <td>Ed Sheeran</td>
      <td>0.461</td>
      <td>0.834</td>
      <td>2.0</td>
      <td>-4.868</td>
      <td>1.0</td>
      <td>0.0989</td>
      <td>0.023200</td>
      <td>0.000011</td>
      <td>0.1400</td>
      <td>0.4710</td>
      <td>135.007</td>
      <td>261154.0</td>
      <td>4.0</td>
      <td>83.4</td>
      <td>D</td>
    </tr>
    <tr>
      <th>18</th>
      <td>5knuzwU65gJK7IF5yJsua</td>
      <td>Rockabye (feat. Sean Paul &amp; Anne-Marie)</td>
      <td>Clean Bandit</td>
      <td>0.720</td>
      <td>0.763</td>
      <td>9.0</td>
      <td>-4.068</td>
      <td>0.0</td>
      <td>0.0523</td>
      <td>0.406000</td>
      <td>0.000000</td>
      <td>0.1800</td>
      <td>0.7420</td>
      <td>101.965</td>
      <td>251088.0</td>
      <td>4.0</td>
      <td>76.3</td>
      <td>A</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0CcQNd8CINkwQfe1RDtGV</td>
      <td>Believer</td>
      <td>Imagine Dragons</td>
      <td>0.779</td>
      <td>0.787</td>
      <td>10.0</td>
      <td>-4.305</td>
      <td>0.0</td>
      <td>0.1080</td>
      <td>0.052400</td>
      <td>0.000000</td>
      <td>0.1400</td>
      <td>0.7080</td>
      <td>124.982</td>
      <td>204347.0</td>
      <td>4.0</td>
      <td>78.7</td>
      <td>Bb</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2rb5MvYT7ZIxbKW5hfcHx</td>
      <td>Mi Gente</td>
      <td>J Balvin</td>
      <td>0.543</td>
      <td>0.677</td>
      <td>11.0</td>
      <td>-4.915</td>
      <td>0.0</td>
      <td>0.0993</td>
      <td>0.014800</td>
      <td>0.000006</td>
      <td>0.1300</td>
      <td>0.2940</td>
      <td>103.809</td>
      <td>189440.0</td>
      <td>4.0</td>
      <td>67.7</td>
      <td>B</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0tKcYR2II1VCQWT79i5Nr</td>
      <td>Thunder</td>
      <td>Imagine Dragons</td>
      <td>0.600</td>
      <td>0.810</td>
      <td>0.0</td>
      <td>-4.749</td>
      <td>1.0</td>
      <td>0.0479</td>
      <td>0.006830</td>
      <td>0.210000</td>
      <td>0.1550</td>
      <td>0.2980</td>
      <td>167.880</td>
      <td>187147.0</td>
      <td>4.0</td>
      <td>81.0</td>
      <td>C</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5uCax9HTNlzGybIStD3vD</td>
      <td>Say You Won't Let Go</td>
      <td>James Arthur</td>
      <td>0.358</td>
      <td>0.557</td>
      <td>10.0</td>
      <td>-7.398</td>
      <td>1.0</td>
      <td>0.0590</td>
      <td>0.695000</td>
      <td>0.000000</td>
      <td>0.0902</td>
      <td>0.4940</td>
      <td>85.043</td>
      <td>211467.0</td>
      <td>4.0</td>
      <td>55.7</td>
      <td>Bb</td>
    </tr>
    <tr>
      <th>23</th>
      <td>79cuOz3SPQTuFrp8WgftA</td>
      <td>There's Nothing Holdin' Me Back</td>
      <td>Shawn Mendes</td>
      <td>0.857</td>
      <td>0.800</td>
      <td>2.0</td>
      <td>-4.035</td>
      <td>1.0</td>
      <td>0.0583</td>
      <td>0.381000</td>
      <td>0.000000</td>
      <td>0.0913</td>
      <td>0.9660</td>
      <td>121.996</td>
      <td>199440.0</td>
      <td>4.0</td>
      <td>80.0</td>
      <td>D</td>
    </tr>
    <tr>
      <th>24</th>
      <td>6De0lHrwBfPfrhorm9q1X</td>
      <td>Me Rehúso</td>
      <td>Danny Ocean</td>
      <td>0.744</td>
      <td>0.804</td>
      <td>1.0</td>
      <td>-6.327</td>
      <td>1.0</td>
      <td>0.0677</td>
      <td>0.023100</td>
      <td>0.000000</td>
      <td>0.0494</td>
      <td>0.4260</td>
      <td>104.823</td>
      <td>205715.0</td>
      <td>4.0</td>
      <td>80.4</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>25</th>
      <td>6D0b04NJIKfEMg040WioJ</td>
      <td>Issues</td>
      <td>Julia Michaels</td>
      <td>0.706</td>
      <td>0.427</td>
      <td>8.0</td>
      <td>-6.864</td>
      <td>1.0</td>
      <td>0.0879</td>
      <td>0.413000</td>
      <td>0.000000</td>
      <td>0.0609</td>
      <td>0.4200</td>
      <td>113.804</td>
      <td>176320.0</td>
      <td>4.0</td>
      <td>42.7</td>
      <td>Ab</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0afhq8XCExXpqazXczTSv</td>
      <td>Galway Girl</td>
      <td>Ed Sheeran</td>
      <td>0.624</td>
      <td>0.876</td>
      <td>9.0</td>
      <td>-3.374</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.073500</td>
      <td>0.000000</td>
      <td>0.3270</td>
      <td>0.7810</td>
      <td>99.943</td>
      <td>170827.0</td>
      <td>4.0</td>
      <td>87.6</td>
      <td>A</td>
    </tr>
    <tr>
      <th>27</th>
      <td>3ebXMykcMXOcLeJ9xZ17X</td>
      <td>Scared to Be Lonely</td>
      <td>Martin Garrix</td>
      <td>0.584</td>
      <td>0.540</td>
      <td>1.0</td>
      <td>-7.786</td>
      <td>0.0</td>
      <td>0.0576</td>
      <td>0.089500</td>
      <td>0.000000</td>
      <td>0.2610</td>
      <td>0.1950</td>
      <td>137.972</td>
      <td>220883.0</td>
      <td>4.0</td>
      <td>54.0</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>28</th>
      <td>7BKLCZ1jbUBVqRi2FVlTV</td>
      <td>Closer</td>
      <td>The Chainsmokers</td>
      <td>0.748</td>
      <td>0.524</td>
      <td>8.0</td>
      <td>-5.599</td>
      <td>1.0</td>
      <td>0.0338</td>
      <td>0.414000</td>
      <td>0.000000</td>
      <td>0.1110</td>
      <td>0.6610</td>
      <td>95.010</td>
      <td>244960.0</td>
      <td>4.0</td>
      <td>52.4</td>
      <td>Ab</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1x5sYLZiu9r5E43kMlt9f</td>
      <td>Symphony (feat. Zara Larsson)</td>
      <td>Clean Bandit</td>
      <td>0.707</td>
      <td>0.629</td>
      <td>0.0</td>
      <td>-4.581</td>
      <td>0.0</td>
      <td>0.0563</td>
      <td>0.259000</td>
      <td>0.000016</td>
      <td>0.1380</td>
      <td>0.4570</td>
      <td>122.863</td>
      <td>212459.0</td>
      <td>4.0</td>
      <td>62.9</td>
      <td>C</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5Ohxk2dO5COHF1krpoPig</td>
      <td>Sign of the Times</td>
      <td>Harry Styles</td>
      <td>0.516</td>
      <td>0.595</td>
      <td>5.0</td>
      <td>-4.630</td>
      <td>1.0</td>
      <td>0.0313</td>
      <td>0.027500</td>
      <td>0.000000</td>
      <td>0.1090</td>
      <td>0.2220</td>
      <td>119.972</td>
      <td>340707.0</td>
      <td>4.0</td>
      <td>59.5</td>
      <td>F</td>
    </tr>
    <tr>
      <th>71</th>
      <td>6gBFPUFcJLzWGx4lenP6h</td>
      <td>goosebumps</td>
      <td>Travis Scott</td>
      <td>0.841</td>
      <td>0.728</td>
      <td>7.0</td>
      <td>-3.370</td>
      <td>1.0</td>
      <td>0.0484</td>
      <td>0.084700</td>
      <td>0.000000</td>
      <td>0.1490</td>
      <td>0.4300</td>
      <td>130.049</td>
      <td>243837.0</td>
      <td>4.0</td>
      <td>72.8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>72</th>
      <td>5Z3GHaZ6ec9bsiI5Benrb</td>
      <td>Young Dumb &amp; Broke</td>
      <td>Khalid</td>
      <td>0.798</td>
      <td>0.539</td>
      <td>1.0</td>
      <td>-6.351</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.199000</td>
      <td>0.000017</td>
      <td>0.1650</td>
      <td>0.3940</td>
      <td>136.949</td>
      <td>202547.0</td>
      <td>4.0</td>
      <td>53.9</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>73</th>
      <td>6jA8HL9i4QGzsj6fjoxp8</td>
      <td>There for You</td>
      <td>Martin Garrix</td>
      <td>0.611</td>
      <td>0.644</td>
      <td>6.0</td>
      <td>-7.607</td>
      <td>0.0</td>
      <td>0.0553</td>
      <td>0.124000</td>
      <td>0.000000</td>
      <td>0.1240</td>
      <td>0.1300</td>
      <td>105.969</td>
      <td>221904.0</td>
      <td>4.0</td>
      <td>64.4</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>74</th>
      <td>21TdkDRXuAB3k90ujRU1e</td>
      <td>Cold (feat. Future)</td>
      <td>Maroon 5</td>
      <td>0.697</td>
      <td>0.716</td>
      <td>9.0</td>
      <td>-6.288</td>
      <td>0.0</td>
      <td>0.1130</td>
      <td>0.118000</td>
      <td>0.000000</td>
      <td>0.0424</td>
      <td>0.5060</td>
      <td>99.905</td>
      <td>234308.0</td>
      <td>4.0</td>
      <td>71.6</td>
      <td>A</td>
    </tr>
    <tr>
      <th>75</th>
      <td>7vGuf3Y35N4wmASOKLUVV</td>
      <td>Silence</td>
      <td>Marshmello</td>
      <td>0.520</td>
      <td>0.761</td>
      <td>4.0</td>
      <td>-3.093</td>
      <td>1.0</td>
      <td>0.0853</td>
      <td>0.256000</td>
      <td>0.000005</td>
      <td>0.1700</td>
      <td>0.2860</td>
      <td>141.971</td>
      <td>180823.0</td>
      <td>4.0</td>
      <td>76.1</td>
      <td>E</td>
    </tr>
    <tr>
      <th>76</th>
      <td>1mXVgsBdtIVeCLJnSnmtd</td>
      <td>Too Good At Goodbyes</td>
      <td>Sam Smith</td>
      <td>0.698</td>
      <td>0.375</td>
      <td>5.0</td>
      <td>-8.279</td>
      <td>1.0</td>
      <td>0.0491</td>
      <td>0.652000</td>
      <td>0.000000</td>
      <td>0.1730</td>
      <td>0.5340</td>
      <td>91.920</td>
      <td>201000.0</td>
      <td>4.0</td>
      <td>37.5</td>
      <td>F</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3EmmCZoqpWOTY1g2GBwJo</td>
      <td>Just Hold On</td>
      <td>Steve Aoki</td>
      <td>0.647</td>
      <td>0.932</td>
      <td>11.0</td>
      <td>-3.515</td>
      <td>1.0</td>
      <td>0.0824</td>
      <td>0.003830</td>
      <td>0.000002</td>
      <td>0.0574</td>
      <td>0.3740</td>
      <td>114.991</td>
      <td>198774.0</td>
      <td>4.0</td>
      <td>93.2</td>
      <td>B</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6uFsE1JgZ20EXyU0JQZbU</td>
      <td>Look What You Made Me Do</td>
      <td>Taylor Swift</td>
      <td>0.773</td>
      <td>0.680</td>
      <td>9.0</td>
      <td>-6.378</td>
      <td>0.0</td>
      <td>0.1410</td>
      <td>0.213000</td>
      <td>0.000016</td>
      <td>0.1220</td>
      <td>0.4970</td>
      <td>128.062</td>
      <td>211859.0</td>
      <td>4.0</td>
      <td>68.0</td>
      <td>A</td>
    </tr>
    <tr>
      <th>79</th>
      <td>0CokSRCu5hZgPxcZBaEzV</td>
      <td>Glorious (feat. Skylar Grey)</td>
      <td>Macklemore</td>
      <td>0.731</td>
      <td>0.794</td>
      <td>0.0</td>
      <td>-5.126</td>
      <td>0.0</td>
      <td>0.0522</td>
      <td>0.032300</td>
      <td>0.000026</td>
      <td>0.1120</td>
      <td>0.3560</td>
      <td>139.994</td>
      <td>220454.0</td>
      <td>4.0</td>
      <td>79.4</td>
      <td>C</td>
    </tr>
    <tr>
      <th>80</th>
      <td>6875MeXyCW0wLyT72Eetm</td>
      <td>Starving</td>
      <td>Hailee Steinfeld</td>
      <td>0.721</td>
      <td>0.626</td>
      <td>4.0</td>
      <td>-4.200</td>
      <td>1.0</td>
      <td>0.1230</td>
      <td>0.402000</td>
      <td>0.000000</td>
      <td>0.1020</td>
      <td>0.5580</td>
      <td>99.914</td>
      <td>181933.0</td>
      <td>4.0</td>
      <td>62.6</td>
      <td>E</td>
    </tr>
    <tr>
      <th>81</th>
      <td>3AEZUABDXNtecAOSC1qTf</td>
      <td>Reggaetón Lento (Bailemos)</td>
      <td>CNCO</td>
      <td>0.761</td>
      <td>0.838</td>
      <td>4.0</td>
      <td>-3.073</td>
      <td>0.0</td>
      <td>0.0502</td>
      <td>0.400000</td>
      <td>0.000000</td>
      <td>0.1760</td>
      <td>0.7100</td>
      <td>93.974</td>
      <td>222560.0</td>
      <td>4.0</td>
      <td>83.8</td>
      <td>E</td>
    </tr>
    <tr>
      <th>82</th>
      <td>3E2Zh20GDCR9B1EYjfXWy</td>
      <td>Weak</td>
      <td>AJR</td>
      <td>0.673</td>
      <td>0.637</td>
      <td>5.0</td>
      <td>-4.518</td>
      <td>1.0</td>
      <td>0.0429</td>
      <td>0.137000</td>
      <td>0.000000</td>
      <td>0.1840</td>
      <td>0.6780</td>
      <td>123.980</td>
      <td>201160.0</td>
      <td>4.0</td>
      <td>63.7</td>
      <td>F</td>
    </tr>
    <tr>
      <th>83</th>
      <td>4pLwZjInHj3SimIyN9SnO</td>
      <td>Side To Side</td>
      <td>Ariana Grande</td>
      <td>0.648</td>
      <td>0.738</td>
      <td>6.0</td>
      <td>-5.883</td>
      <td>0.0</td>
      <td>0.2470</td>
      <td>0.040800</td>
      <td>0.000000</td>
      <td>0.2920</td>
      <td>0.6030</td>
      <td>159.145</td>
      <td>226160.0</td>
      <td>4.0</td>
      <td>73.8</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>84</th>
      <td>3QwBODjSEzelZyVjxPOHd</td>
      <td>Otra Vez (feat. J Balvin)</td>
      <td>Zion &amp; Lennox</td>
      <td>0.832</td>
      <td>0.772</td>
      <td>10.0</td>
      <td>-5.429</td>
      <td>1.0</td>
      <td>0.1000</td>
      <td>0.055900</td>
      <td>0.000486</td>
      <td>0.4400</td>
      <td>0.7040</td>
      <td>96.016</td>
      <td>209453.0</td>
      <td>4.0</td>
      <td>77.2</td>
      <td>Bb</td>
    </tr>
    <tr>
      <th>85</th>
      <td>1wjzFQodRWrPcQ0AnYnvQ</td>
      <td>I Like Me Better</td>
      <td>Lauv</td>
      <td>0.752</td>
      <td>0.505</td>
      <td>9.0</td>
      <td>-7.621</td>
      <td>1.0</td>
      <td>0.2530</td>
      <td>0.535000</td>
      <td>0.000003</td>
      <td>0.1040</td>
      <td>0.4190</td>
      <td>91.970</td>
      <td>197437.0</td>
      <td>4.0</td>
      <td>50.5</td>
      <td>A</td>
    </tr>
    <tr>
      <th>86</th>
      <td>04DwTuZ2VBdJCCC5TROn7</td>
      <td>In the Name of Love</td>
      <td>Martin Garrix</td>
      <td>0.490</td>
      <td>0.485</td>
      <td>4.0</td>
      <td>-6.237</td>
      <td>0.0</td>
      <td>0.0406</td>
      <td>0.059200</td>
      <td>0.000000</td>
      <td>0.3370</td>
      <td>0.1960</td>
      <td>133.889</td>
      <td>195840.0</td>
      <td>4.0</td>
      <td>48.5</td>
      <td>E</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6DNtNfH8hXkqOX1sjqmI7</td>
      <td>Cold Water (feat. Justin Bieber &amp; MØ)</td>
      <td>Major Lazer</td>
      <td>0.608</td>
      <td>0.798</td>
      <td>6.0</td>
      <td>-5.092</td>
      <td>0.0</td>
      <td>0.0432</td>
      <td>0.073600</td>
      <td>0.000000</td>
      <td>0.1560</td>
      <td>0.5010</td>
      <td>92.943</td>
      <td>185352.0</td>
      <td>4.0</td>
      <td>79.8</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>88</th>
      <td>1UZOjK1BwmwWU14Erba9C</td>
      <td>Malibu</td>
      <td>Miley Cyrus</td>
      <td>0.573</td>
      <td>0.781</td>
      <td>8.0</td>
      <td>-6.406</td>
      <td>1.0</td>
      <td>0.0555</td>
      <td>0.076700</td>
      <td>0.000026</td>
      <td>0.0813</td>
      <td>0.3430</td>
      <td>139.934</td>
      <td>231907.0</td>
      <td>4.0</td>
      <td>78.1</td>
      <td>Ab</td>
    </tr>
    <tr>
      <th>89</th>
      <td>4b4KcovePX8Ke2cLIQTLM</td>
      <td>All Night</td>
      <td>The Vamps</td>
      <td>0.544</td>
      <td>0.809</td>
      <td>8.0</td>
      <td>-5.098</td>
      <td>1.0</td>
      <td>0.0363</td>
      <td>0.003800</td>
      <td>0.000000</td>
      <td>0.3230</td>
      <td>0.4480</td>
      <td>145.017</td>
      <td>197640.0</td>
      <td>4.0</td>
      <td>80.9</td>
      <td>Ab</td>
    </tr>
    <tr>
      <th>90</th>
      <td>1a5Yu5L18qNxVhXx38njO</td>
      <td>Hear Me Now</td>
      <td>Alok</td>
      <td>0.789</td>
      <td>0.442</td>
      <td>11.0</td>
      <td>-7.844</td>
      <td>1.0</td>
      <td>0.0421</td>
      <td>0.586000</td>
      <td>0.003660</td>
      <td>0.0927</td>
      <td>0.4500</td>
      <td>121.971</td>
      <td>192846.0</td>
      <td>4.0</td>
      <td>44.2</td>
      <td>B</td>
    </tr>
    <tr>
      <th>91</th>
      <td>4c2W3VKsOFoIg2SFaO6DY</td>
      <td>Your Song</td>
      <td>Rita Ora</td>
      <td>0.855</td>
      <td>0.624</td>
      <td>1.0</td>
      <td>-4.093</td>
      <td>1.0</td>
      <td>0.0488</td>
      <td>0.158000</td>
      <td>0.000000</td>
      <td>0.0513</td>
      <td>0.9620</td>
      <td>117.959</td>
      <td>180757.0</td>
      <td>4.0</td>
      <td>62.4</td>
      <td>C#/Db</td>
    </tr>
    <tr>
      <th>92</th>
      <td>22eADXu8DfOAUEDw4vU8q</td>
      <td>Ahora Dice</td>
      <td>Chris Jeday</td>
      <td>0.708</td>
      <td>0.693</td>
      <td>6.0</td>
      <td>-5.516</td>
      <td>1.0</td>
      <td>0.1380</td>
      <td>0.246000</td>
      <td>0.000000</td>
      <td>0.1290</td>
      <td>0.4270</td>
      <td>143.965</td>
      <td>271080.0</td>
      <td>4.0</td>
      <td>69.3</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>93</th>
      <td>7nZmah2llfvLDiUjm0kiy</td>
      <td>Friends (with BloodPop®)</td>
      <td>Justin Bieber</td>
      <td>0.744</td>
      <td>0.739</td>
      <td>8.0</td>
      <td>-5.350</td>
      <td>1.0</td>
      <td>0.0387</td>
      <td>0.004590</td>
      <td>0.000000</td>
      <td>0.3060</td>
      <td>0.6490</td>
      <td>104.990</td>
      <td>189467.0</td>
      <td>4.0</td>
      <td>73.9</td>
      <td>Ab</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2fQrGHiQOvpL9UgPvtYy6</td>
      <td>Bank Account</td>
      <td>21 Savage</td>
      <td>0.884</td>
      <td>0.346</td>
      <td>8.0</td>
      <td>-8.228</td>
      <td>0.0</td>
      <td>0.3510</td>
      <td>0.015100</td>
      <td>0.000007</td>
      <td>0.0871</td>
      <td>0.3760</td>
      <td>75.016</td>
      <td>220307.0</td>
      <td>4.0</td>
      <td>34.6</td>
      <td>Ab</td>
    </tr>
    <tr>
      <th>95</th>
      <td>1PSBzsahR2AKwLJgx8ehB</td>
      <td>Bad Things (with Camila Cabello)</td>
      <td>Machine Gun Kelly</td>
      <td>0.675</td>
      <td>0.690</td>
      <td>2.0</td>
      <td>-4.761</td>
      <td>1.0</td>
      <td>0.1320</td>
      <td>0.210000</td>
      <td>0.000000</td>
      <td>0.2870</td>
      <td>0.2720</td>
      <td>137.817</td>
      <td>239293.0</td>
      <td>4.0</td>
      <td>69.0</td>
      <td>D</td>
    </tr>
    <tr>
      <th>96</th>
      <td>0QsvXIfqM0zZoerQfsI9l</td>
      <td>Don't Let Me Down</td>
      <td>The Chainsmokers</td>
      <td>0.542</td>
      <td>0.859</td>
      <td>11.0</td>
      <td>-5.651</td>
      <td>1.0</td>
      <td>0.1970</td>
      <td>0.160000</td>
      <td>0.004660</td>
      <td>0.1370</td>
      <td>0.4030</td>
      <td>159.797</td>
      <td>208053.0</td>
      <td>4.0</td>
      <td>85.9</td>
      <td>B</td>
    </tr>
    <tr>
      <th>97</th>
      <td>7mldq42yDuxiUNn08nvzH</td>
      <td>Body Like A Back Road</td>
      <td>Sam Hunt</td>
      <td>0.731</td>
      <td>0.469</td>
      <td>5.0</td>
      <td>-7.226</td>
      <td>1.0</td>
      <td>0.0326</td>
      <td>0.463000</td>
      <td>0.000001</td>
      <td>0.1030</td>
      <td>0.6310</td>
      <td>98.963</td>
      <td>165387.0</td>
      <td>4.0</td>
      <td>46.9</td>
      <td>F</td>
    </tr>
    <tr>
      <th>98</th>
      <td>7i2DJ88J7jQ8K7zqFX2fW</td>
      <td>Now Or Never</td>
      <td>Halsey</td>
      <td>0.658</td>
      <td>0.588</td>
      <td>6.0</td>
      <td>-4.902</td>
      <td>0.0</td>
      <td>0.0367</td>
      <td>0.105000</td>
      <td>0.000001</td>
      <td>0.1250</td>
      <td>0.4340</td>
      <td>110.075</td>
      <td>214802.0</td>
      <td>4.0</td>
      <td>58.8</td>
      <td>F#/Gb</td>
    </tr>
    <tr>
      <th>99</th>
      <td>1j4kHkkpqZRBwE0A4CN4Y</td>
      <td>Dusk Till Dawn - Radio Edit</td>
      <td>ZAYN</td>
      <td>0.258</td>
      <td>0.437</td>
      <td>11.0</td>
      <td>-6.593</td>
      <td>0.0</td>
      <td>0.0390</td>
      <td>0.101000</td>
      <td>0.000001</td>
      <td>0.1060</td>
      <td>0.0967</td>
      <td>180.043</td>
      <td>239000.0</td>
      <td>4.0</td>
      <td>43.7</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 18 columns</p>
</div>



## Energy of Songs- Visualized 4 Ways


```python
x = df_transform.energy
data = [go.Histogram(x=x)]
layout = go.Layout(
    title='Energy of Songs',
    xaxis=dict(
        title='Energy',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number of Songs',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='basic histogram')

```


<div id="06e417b5-328e-44bc-ab18-f01c476a8147" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("06e417b5-328e-44bc-ab18-f01c476a8147", [{"x": [0.652, 0.815, 0.7859999999999999, 0.635, 0.6679999999999999, 0.611, 0.5329999999999999, 0.769, 0.56, 0.451, 0.75, 0.6579999999999999, 0.634, 0.626, 0.434, 0.812, 0.8170000000000001, 0.8340000000000001, 0.763, 0.787, 0.677, 0.81, 0.557, 0.8, 0.804, 0.42700000000000005, 0.8759999999999999, 0.54, 0.524, 0.629, 0.813, 0.594, 0.672, 0.795, 0.696, 0.5720000000000001, 0.46299999999999997, 0.522, 0.485, 0.65, 0.44799999999999995, 0.843, 0.745, 0.7929999999999999, 0.789, 0.653, 0.449, 0.773, 0.665, 0.517, 0.836, 0.488, 0.718, 0.743, 0.619, 0.823, 0.868, 0.633, 0.359, 0.8029999999999999, 0.514, 0.691, 0.669, 0.8009999999999999, 0.667, 0.617, 0.555, 0.418, 0.8640000000000001, 0.5670000000000001, 0.595, 0.728, 0.539, 0.644, 0.716, 0.7609999999999999, 0.375, 0.932, 0.68, 0.794, 0.626, 0.838, 0.637, 0.738, 0.772, 0.505, 0.485, 0.7979999999999999, 0.7809999999999999, 0.809, 0.442, 0.624, 0.693, 0.7390000000000001, 0.34600000000000003, 0.69, 0.8590000000000001, 0.469, 0.588, 0.43700000000000006], "type": "histogram", "uid": "4851a6cc-aa4f-11e8-ab49-9a0015cd93e0"}], {"title": "Energy of Songs", "xaxis": {"title": "Energy", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Number of Songs", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
data = [go.Bar(x=df_transform.name,
            y=df_transform.energy)]
layout = go.Layout(
    title='Energy by Song',
    xaxis=dict(
        title='Name',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Energyy',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='basic-bar')
```


<div id="d4967775-bce6-427a-bab7-ed3c3e47449d" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("d4967775-bce6-427a-bab7-ed3c3e47449d", [{"x": ["Shape of You", "Despacito - Remix", "Despacito (Featuring Daddy Yankee)", "Something Just Like This", "I'm the One", "HUMBLE.", "It Ain't Me (with Selena Gomez)", "Unforgettable", "That's What I Like", "I Don\u2019t Wanna Live Forever (Fifty Shades Darker) - From \"Fifty Shades Darker (Original Motion Picture Soundtrack)\"", "XO TOUR Llif3", "Paris", "Stay (with Alessia Cara)", "Attention", "Mask Off", "Congratulations", "Swalla (feat. Nicki Minaj & Ty Dolla $ign)", "Castle on the Hill", "Rockabye (feat. Sean Paul & Anne-Marie)", "Believer", "Mi Gente", "Thunder", "Say You Won't Let Go", "There's Nothing Holdin' Me Back", "Me Reh\u00faso", "Issues", "Galway Girl", "Scared to Be Lonely", "Closer", "Symphony (feat. Zara Larsson)", "I Feel It Coming", "Starboy", "Wild Thoughts", "Slide", "New Rules", "1-800-273-8255", "Passionfruit", "rockstar", "Strip That Down", "2U (feat. Justin Bieber)", "Perfect", "Call On Me - Ryan Riback Extended Remix", "Feels", "Mama", "Felices los 4", "iSpy (feat. Lil Yachty)", "Location", "Chantaje", "Bad and Boujee (feat. Lil Uzi Vert)", "Havana", "Solo Dance", "Fake Love", "Let Me Love You", "More Than You Know", "One Dance", "SUBEME LA RADIO", "Pretty Girl - Cheat Codes X CADE Remix", "Sorry Not Sorry", "Redbone", "24K Magic", "DNA.", "El Amante", "You Don't Know Me - Radio Edit", "Chained To The Rhythm", "No Promises (feat. Demi Lovato)", "Don't Wanna Know (feat. Kendrick Lamar)", "How Far I'll Go - From \"Moana\"", "Slow Hands", "Esc\u00e1pate Conmigo", "Bounce Back", "Sign of the Times", "goosebumps", "Young Dumb & Broke", "There for You", "Cold (feat. Future)", "Silence", "Too Good At Goodbyes", "Just Hold On", "Look What You Made Me Do", "Glorious (feat. Skylar Grey)", "Starving", "Reggaet\u00f3n Lento (Bailemos)", "Weak", "Side To Side", "Otra Vez (feat. J Balvin)", "I Like Me Better", "In the Name of Love", "Cold Water (feat. Justin Bieber & M\u00d8)", "Malibu", "All Night", "Hear Me Now", "Your Song", "Ahora Dice", "Friends (with BloodPop\u00ae)", "Bank Account", "Bad Things (with Camila Cabello)", "Don't Let Me Down", "Body Like A Back Road", "Now Or Never", "Dusk Till Dawn - Radio Edit"], "y": [0.652, 0.815, 0.7859999999999999, 0.635, 0.6679999999999999, 0.611, 0.5329999999999999, 0.769, 0.56, 0.451, 0.75, 0.6579999999999999, 0.634, 0.626, 0.434, 0.812, 0.8170000000000001, 0.8340000000000001, 0.763, 0.787, 0.677, 0.81, 0.557, 0.8, 0.804, 0.42700000000000005, 0.8759999999999999, 0.54, 0.524, 0.629, 0.813, 0.594, 0.672, 0.795, 0.696, 0.5720000000000001, 0.46299999999999997, 0.522, 0.485, 0.65, 0.44799999999999995, 0.843, 0.745, 0.7929999999999999, 0.789, 0.653, 0.449, 0.773, 0.665, 0.517, 0.836, 0.488, 0.718, 0.743, 0.619, 0.823, 0.868, 0.633, 0.359, 0.8029999999999999, 0.514, 0.691, 0.669, 0.8009999999999999, 0.667, 0.617, 0.555, 0.418, 0.8640000000000001, 0.5670000000000001, 0.595, 0.728, 0.539, 0.644, 0.716, 0.7609999999999999, 0.375, 0.932, 0.68, 0.794, 0.626, 0.838, 0.637, 0.738, 0.772, 0.505, 0.485, 0.7979999999999999, 0.7809999999999999, 0.809, 0.442, 0.624, 0.693, 0.7390000000000001, 0.34600000000000003, 0.69, 0.8590000000000001, 0.469, 0.588, 0.43700000000000006], "type": "bar", "uid": "92db2bc2-abd6-11e8-8e9e-9a0015cd93e0"}], {"title": "Energy by Song", "xaxis": {"title": "Name", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Energyy", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
trace1 = go.Scatter(
    y = df_transform.energy,
    mode='markers',
    marker=dict(
        size=16,
        color = np.random.randn(500), #set color equal to a variable
        colorscale='Viridis',
        showscale=True
    )
)
data = [trace1]
layout = go.Layout(
    title='Songs',
    xaxis=dict(
        title='Energy',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Energy',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)
fig = go.Figure(data=data, layout=layout)


iplot(fig, filename='scatter-plot-with-colorscale')
```


<div id="7fe07b19-2b3d-4f26-8e8e-317d2060a2ed" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
        Plotly.plot(
            '7fe07b19-2b3d-4f26-8e8e-317d2060a2ed',
            [{"marker": {"color": [0.04944805363351229, -0.4307880737660511, -0.18807962990494428, -1.9959453216428482, 0.610036115214979, -0.31913714144843425, -0.6518590833453838, 0.9441442194475557, 0.4459513672133308, -0.15211442574669662, 1.5068628592573126, 0.9141626184320405, 1.3289084837200276, -0.028552508411588077, -0.6503608545137902, 0.885783213971398, -1.0241078075155985, -0.5216329447526965, -0.04346695297564423, 0.11081733284521883, 0.3041898103066505, -0.37614266899699556, 0.29069087179748043, -0.2153970504569113, -1.146864047821728, 0.8187874348446347, -1.4921366051570866, -0.5751229551208994, -0.7882346547881856, 0.3101449001481461, 0.15805174102925615, -1.8113426400700268, 0.40956835885976417, -0.17737267107229432, -0.9371445877480313, -0.5138599530166122, 1.0860203742070598, 0.41527391026621596, -0.42680555997315717, -1.267886000683359, 1.5008844515168156, -0.18869615011155652, 0.7865455486250383, 1.134757267659026, 0.3696615167309504, -1.862930734392036, -0.3500597724544794, -0.8332897956047061, -0.17647494429685323, 0.5180877264796327, -0.9514863159651298, 0.7181966480301804, 0.026838491653022087, -0.6100489220518175, -0.4327923910985015, 1.2111645593430558, -0.8969573377671939, -1.0504752726072004, 1.7445300518551639, 0.03570067347424023, -0.7447149744936289, 0.7203459567440617, -1.194832402369368, -0.6548622349406411, 1.241052644479894, 0.9225889981608566, 0.01945657197779063, -1.2600263314908253, 0.28281739381092397, -0.9201188027988882, -0.6405336578626378, -1.1400991418053485, 0.19151262530858434, -1.7638933190911825, -0.713811023224731, 0.34857357962569746, 0.5961239441000201, 0.5923649653454971, 0.4982725934806361, -1.53427952545271, 1.0046286342009805, -1.736950361168324, 1.5231085113181806, 1.6391385243237149, -0.6294666095786219, -0.34925870666230846, 0.9837798875057314, -0.688234270457065, 0.5081122717138522, 1.6431542017061653, -0.5153220885791364, -0.43932163891950377, -1.501270496695997, 0.951456898160257, -0.3928982171947117, 0.34631387889875265, -0.1813402933956491, 0.2718811242258912, 0.8970760663898325, 0.2082661290991015, 0.22660372663724576, -0.12494509391355883, 0.096719394923831, 0.126572343015574, -0.18769698603157642, -0.8744125576793521, 0.8918693775251143, 1.1242694012616365, -1.9709331764185878, -0.39806055766120385, 1.7661765557833407, -1.1446369805517551, 1.5553930464311045, 0.18492294468300444, 1.5379088791862041, -0.8670947479468422, -0.2336102936227536, 1.2753039104281656, 0.6883765264720744, -1.5652509511422634, 0.241602561792021, -1.1273708747182158, -0.1842044080542373, -0.13212716891962392, 0.9894338208721596, -0.843200802821387, 0.3422686034842848, -0.5188641044123778, 0.0756190804761075, 0.8790449670563488, -0.16208324817593334, -0.756121585897653, 1.1635695827404486, 0.7355681153436028, -1.071091025247739, -0.4420825190408012, -0.056771911737940946, -0.07478708593017162, 0.9339774285658426, -2.4530805609802306, -0.6918744873064442, -1.0942278445600144, -0.27306176491920414, -0.33712141494422565, -0.8368465915187526, 0.2797232773701399, -2.2253852041804545, 0.25247955667134026, -0.5691041849634652, -0.44005536959556607, -1.1529915221056268, -0.033391286315339296, 0.014059933055176985, 0.4075093141701329, -1.1187346506094809, -0.5938585619572021, -0.484381834002389, -0.9073424247332451, 0.31213701297593777, 0.542869266050468, -0.0787286488350874, 0.3830100404318694, 0.5583145785933487, -1.1761642196928312, -0.2306072245858031, -0.7562522046455016, -1.2101333318172185, 0.6618685131012336, 0.8982570553926421, 0.9074166991841809, -1.2361416414340498, 0.4440621853098306, -0.06731057112791267, 0.04358963356404376, 0.40642261671527213, 1.4859629993400596, -0.0005003748287525471, 0.3208163375798361, -2.4249013789436806, 0.9538450545387351, 1.7348038403673858, -0.18630041552247142, -0.2744015294528235, 0.7781237016644292, 0.09729906977735507, -1.1318495942759994, -0.8554382819394122, -0.32747795877696845, 1.2767075322224553, 1.0444767803860653, 0.8432251377144138, 0.9575936495705137, 0.30117948035167164, -1.6972349101912938, 0.1291924346287843, -0.20061764402195623, 1.4344989833949309, 0.8523342491215283, -0.691714075813393, 0.2422881028558278, -0.4722159234320944, 0.1957261764524678, 1.1102815206108445, -1.4651302750389663, 0.3577948538332536, -1.3543802848252007, -0.19146894141580356, -0.8812697032395779, -0.5979143785450459, -0.9155512095395207, -1.1298236912077428, 0.3785829752915379, -0.5106159871573813, -1.2679545267649959, -0.09971474066820472, 0.5634965939197083, 0.6444989127652538, -0.1958876241492176, -0.295203319022431, -0.004002261123150513, -0.6259355089448093, -0.3148142441069618, -1.5254138250588831, -0.009130493379551127, 1.7574260686670156, 0.6480953694551772, -0.2957839180408571, -0.7606464574619135, -1.3763864416672469, -0.4149420015179324, 0.24610053531094986, 1.2263364250583673, 0.7673936958462494, -0.662682961830187, -0.07776505620762454, 1.7837861121837755, 0.5284125355704369, 0.915766905281018, 0.5616310160140793, -0.17834265002121813, -0.8367029889529014, -0.16355771721227286, -0.5259578597405477, 0.15598081934922792, 1.518681363626497, 1.0684872489356547, 1.8000415753870354, 0.8349612242163876, 0.2640668054705448, 0.8566849917922047, 1.2370015936917507, 2.9418689955589152, 0.17321105652644017, 0.5754611320799642, 1.7546876576347408, -0.6389101723767253, -0.00037484813343891606, 1.0942170635931696, 0.5765373552608494, -1.0752484318639246, 1.4785399992396218, 0.36915923643479565, -0.7059037145700555, 0.3478609445865731, 0.3188748703702512, -0.0973217022119698, -0.8990587301019396, 0.821826229481226, 1.2851352079778577, 0.00875012563372653, -2.9426356858110974, 0.8581141402757002, -0.10918846951098614, -0.05684330763267216, -0.7819103578663801, 0.12167332544539805, 0.6325944320661868, -1.4474054423010871, -0.65777024996723, 0.16339279935852985, 1.1737186911008242, -0.8455203137023025, -0.22228455470515204, -1.7201575139649647, 0.14642850637312138, 0.8106749921296038, -1.2334510425296992, -1.1621932084226334, -0.29072801665841114, -0.8958707911092504, -0.8467684279209148, 0.9595179371988883, -0.7551629990024549, -1.2880759489816511, 0.9586527089593564, -1.5433225070741015, -2.1413382161300016, -0.09027936170245929, -1.1038941987624817, -1.6878970772402195, -0.2177368338652421, 0.7695228511157467, 1.018844214155944, -0.17949122167888132, 0.1483124358457029, -1.9313082256584355, -0.305805053237423, 0.0756096894545282, 0.30571370309602436, -0.36672467962466554, 0.3255300753988932, -0.217030900805645, 0.2036137754231392, -0.5679032847842218, 2.218951828507548, 0.7085851276012171, -2.169553063707022, -0.8707946432466664, -0.27715930277157663, -0.5127958573284017, 0.963219955369038, -0.9396432814478958, -0.5438472171434516, 1.779733677738124, 0.577335577630255, -0.4082091008666094, -0.5151288357588308, -0.9584337775071329, 0.03600023020705212, -0.9750398810804802, -0.6759557561241283, -0.7990569481507908, 1.1705202930603933, 0.5071562174470725, 1.1967477953318373, 0.4601821553748364, -1.2794406939626866, 1.4485859236669458, -1.0949404680516006, -0.5218868462009605, -1.6660088519138825, 2.1062155240296745, -0.5347416919739412, 0.1586244739972903, -0.5454374225081718, 0.5827275729900763, 0.8745241829881215, -1.0181210140836645, 2.4875425584337436, 0.4106183706291976, -0.8174596478367623, 0.8563132573395881, 0.26972042945329083, 0.7972916778434299, -0.6496354628665181, -0.14603100721859505, 0.17173891257595789, -1.0868648763030935, -0.3112182237785842, 0.5886333680396311, -0.2954813413644969, -1.505905121769219, -1.505092841587295, 0.8473006404281451, 0.9885472603778577, 0.5870117829485508, -1.0333113584605287, 0.4124756633439853, 1.3577255556275247, -1.6768574623314685, 0.2832253318021972, -1.3655410389706306, -1.3699134865606777, 0.4224856542723325, -0.3927850190557663, 1.500228445830205, -1.0158275204943208, -1.5315474118402719, -0.23471607521018725, 2.5282281753860527, -0.6134826749054701, 0.4484288901146634, 0.2240338279541646, -0.24932082227296515, -1.4477317955205065, -0.1752015366602657, -0.1980224009591785, -1.5673229086068357, 0.23597662771162214, 0.831710221092441, 3.4148939742665445, 0.06719489567478143, 0.32240309762143393, -2.208729698465835, -1.4541528699067836, -0.3442510582351793, 0.6505620580723847, -1.1233015827931982, -0.8001542824639637, -1.9135132617601207, 0.2385366020583845, 2.1893533717597635, 0.16176487986662302, 1.3921135852285613, -2.956387531798605, -0.6607168378600375, 1.8712175518938352, 0.7502059815494678, -0.14633349399633963, 0.5002670097119447, -0.2622915867638824, 0.3168568949565607, -0.2631745653758101, 0.38826382270252074, 0.15182685005720628, 1.2907230280041013, -0.7649470139730645, -0.07389516822661893, 0.5095464854541174, 0.8969281598150758, 0.8799587470090762, -1.229895655126573, 0.23641332752215732, 1.1027781830728614, -0.9882634576650043, 0.7456064769344226, 0.12511687252615164, 0.2096310235768428, -0.09216206792972377, 1.636381493640163, 1.010324693666727, 0.4553911432446181, 0.8658587279483758, 1.095501755709672, 1.6477382845285935, 0.17563652666883856, 0.05750690702109161, 1.3039157397823844, -0.8793294127310897, -1.0151360042462985, -0.07775661346291604, -1.3422277438532575, -0.05942378852052098, -2.0860847400734, 1.7551957682061858, 0.6369579424219668, -1.9914900795613, -1.8575097992626868, -0.241117759774903, -1.3540906568458033, 1.5186502599459633, 0.592874199588919, 1.9821918611492164, -0.6659042629362586, 0.7142007181485179, 0.45404173592420116, -1.0104964434056098, 0.9966781558121349, -0.6429187646273029, -0.08571216935295213, -1.1538609406679634, -0.5899071866934964, -1.7851464459901842, 1.8813348050100758, -0.8255570431035868, 0.06707226339209808, 0.4133588916413776, -1.1002669870381037, -0.9764211266321915, 0.6759687735481955, -0.05761777449641032, -1.5280956959368976, -0.2945136679416762, -0.19061426923172514, -1.3973357266406656, -2.530139896406591, 0.2762430046039129, -0.9012735602731846, 1.5387062310344155, -1.1883025171496893, 1.6135644947274845, -0.23724570538140138, 0.2164708584896699, -0.15653316938998268, -0.44499103342848273, -0.2197363558885641, -0.02332172747544614, -0.3139943211871182, 0.4690460937848225, -1.0376807922871947, 0.044574743996831305, 0.041469676054459254, 1.7931830808607303, -0.3952561052022087, -0.20501495121195984, -1.0322464209116853, 0.3009969800208442, -0.856827407588094, 0.10008644889380798, 0.4516985748895597], "colorscale": "Viridis", "showscale": true, "size": 16}, "mode": "markers", "y": [0.652, 0.815, 0.7859999999999999, 0.635, 0.6679999999999999, 0.611, 0.5329999999999999, 0.769, 0.56, 0.451, 0.75, 0.6579999999999999, 0.634, 0.626, 0.434, 0.812, 0.8170000000000001, 0.8340000000000001, 0.763, 0.787, 0.677, 0.81, 0.557, 0.8, 0.804, 0.42700000000000005, 0.8759999999999999, 0.54, 0.524, 0.629, 0.813, 0.594, 0.672, 0.795, 0.696, 0.5720000000000001, 0.46299999999999997, 0.522, 0.485, 0.65, 0.44799999999999995, 0.843, 0.745, 0.7929999999999999, 0.789, 0.653, 0.449, 0.773, 0.665, 0.517, 0.836, 0.488, 0.718, 0.743, 0.619, 0.823, 0.868, 0.633, 0.359, 0.8029999999999999, 0.514, 0.691, 0.669, 0.8009999999999999, 0.667, 0.617, 0.555, 0.418, 0.8640000000000001, 0.5670000000000001, 0.595, 0.728, 0.539, 0.644, 0.716, 0.7609999999999999, 0.375, 0.932, 0.68, 0.794, 0.626, 0.838, 0.637, 0.738, 0.772, 0.505, 0.485, 0.7979999999999999, 0.7809999999999999, 0.809, 0.442, 0.624, 0.693, 0.7390000000000001, 0.34600000000000003, 0.69, 0.8590000000000001, 0.469, 0.588, 0.43700000000000006], "type": "scatter", "uid": "f40aa6f4-ab20-11e8-8b17-9a0015cd93e0"}],
            {"title": "Songs", "xaxis": {"title": "Energy", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Energy", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}},
            {"showLink": true, "linkText": "Export to plot.ly"}
        ).then(function () {return Plotly.addFrames('7fe07b19-2b3d-4f26-8e8e-317d2060a2ed',{});}).then(function(){Plotly.animate('7fe07b19-2b3d-4f26-8e8e-317d2060a2ed');})
        });</script>



```python
trace0 = go.Scatter(
    x=df_transform.name,
    y=df_transform.energy_percent,
    mode='markers',
    marker=dict(
        size=df_transform.energy_percent,
    )
)

data = [trace0]
iplot(data, filename='bubblechart-size')
```


<div id="f3ffb5e6-6455-4b72-9ef3-bd28f9b75905" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("f3ffb5e6-6455-4b72-9ef3-bd28f9b75905", [{"marker": {"size": [65.2, 81.5, 78.6, 63.5, 66.8, 61.1, 53.29999999999999, 76.9, 56.00000000000001, 45.1, 75.0, 65.8, 63.4, 62.6, 43.4, 81.2, 81.7, 83.4, 76.3, 78.7, 67.7, 81.0, 55.7, 80.0, 80.4, 42.7, 87.6, 54.0, 52.400000000000006, 62.9, 81.3, 59.4, 67.2, 79.5, 69.6, 57.2, 46.3, 52.2, 48.5, 65.0, 44.8, 84.3, 74.5, 79.3, 78.9, 65.3, 44.9, 77.3, 66.5, 51.7, 83.6, 48.8, 71.8, 74.3, 61.9, 82.3, 86.8, 63.3, 35.9, 80.3, 51.4, 69.1, 66.9, 80.1, 66.7, 61.7, 55.50000000000001, 41.8, 86.4, 56.7, 59.5, 72.8, 53.900000000000006, 64.4, 71.6, 76.1, 37.5, 93.2, 68.0, 79.4, 62.6, 83.8, 63.7, 73.8, 77.2, 50.5, 48.5, 79.8, 78.1, 80.9, 44.2, 62.4, 69.3, 73.9, 34.6, 69.0, 85.9, 46.9, 58.8, 43.7]}, "mode": "markers", "x": ["Shape of You", "Despacito - Remix", "Despacito (Featuring Daddy Yankee)", "Something Just Like This", "I'm the One", "HUMBLE.", "It Ain't Me (with Selena Gomez)", "Unforgettable", "That's What I Like", "I Don\u2019t Wanna Live Forever (Fifty Shades Darker) - From \"Fifty Shades Darker (Original Motion Picture Soundtrack)\"", "XO TOUR Llif3", "Paris", "Stay (with Alessia Cara)", "Attention", "Mask Off", "Congratulations", "Swalla (feat. Nicki Minaj & Ty Dolla $ign)", "Castle on the Hill", "Rockabye (feat. Sean Paul & Anne-Marie)", "Believer", "Mi Gente", "Thunder", "Say You Won't Let Go", "There's Nothing Holdin' Me Back", "Me Reh\u00faso", "Issues", "Galway Girl", "Scared to Be Lonely", "Closer", "Symphony (feat. Zara Larsson)", "I Feel It Coming", "Starboy", "Wild Thoughts", "Slide", "New Rules", "1-800-273-8255", "Passionfruit", "rockstar", "Strip That Down", "2U (feat. Justin Bieber)", "Perfect", "Call On Me - Ryan Riback Extended Remix", "Feels", "Mama", "Felices los 4", "iSpy (feat. Lil Yachty)", "Location", "Chantaje", "Bad and Boujee (feat. Lil Uzi Vert)", "Havana", "Solo Dance", "Fake Love", "Let Me Love You", "More Than You Know", "One Dance", "SUBEME LA RADIO", "Pretty Girl - Cheat Codes X CADE Remix", "Sorry Not Sorry", "Redbone", "24K Magic", "DNA.", "El Amante", "You Don't Know Me - Radio Edit", "Chained To The Rhythm", "No Promises (feat. Demi Lovato)", "Don't Wanna Know (feat. Kendrick Lamar)", "How Far I'll Go - From \"Moana\"", "Slow Hands", "Esc\u00e1pate Conmigo", "Bounce Back", "Sign of the Times", "goosebumps", "Young Dumb & Broke", "There for You", "Cold (feat. Future)", "Silence", "Too Good At Goodbyes", "Just Hold On", "Look What You Made Me Do", "Glorious (feat. Skylar Grey)", "Starving", "Reggaet\u00f3n Lento (Bailemos)", "Weak", "Side To Side", "Otra Vez (feat. J Balvin)", "I Like Me Better", "In the Name of Love", "Cold Water (feat. Justin Bieber & M\u00d8)", "Malibu", "All Night", "Hear Me Now", "Your Song", "Ahora Dice", "Friends (with BloodPop\u00ae)", "Bank Account", "Bad Things (with Camila Cabello)", "Don't Let Me Down", "Body Like A Back Road", "Now Or Never", "Dusk Till Dawn - Radio Edit"], "y": [65.2, 81.5, 78.6, 63.5, 66.8, 61.1, 53.29999999999999, 76.9, 56.00000000000001, 45.1, 75.0, 65.8, 63.4, 62.6, 43.4, 81.2, 81.7, 83.4, 76.3, 78.7, 67.7, 81.0, 55.7, 80.0, 80.4, 42.7, 87.6, 54.0, 52.400000000000006, 62.9, 81.3, 59.4, 67.2, 79.5, 69.6, 57.2, 46.3, 52.2, 48.5, 65.0, 44.8, 84.3, 74.5, 79.3, 78.9, 65.3, 44.9, 77.3, 66.5, 51.7, 83.6, 48.8, 71.8, 74.3, 61.9, 82.3, 86.8, 63.3, 35.9, 80.3, 51.4, 69.1, 66.9, 80.1, 66.7, 61.7, 55.50000000000001, 41.8, 86.4, 56.7, 59.5, 72.8, 53.900000000000006, 64.4, 71.6, 76.1, 37.5, 93.2, 68.0, 79.4, 62.6, 83.8, 63.7, 73.8, 77.2, 50.5, 48.5, 79.8, 78.1, 80.9, 44.2, 62.4, 69.3, 73.9, 34.6, 69.0, 85.9, 46.9, 58.8, 43.7], "type": "scatter", "uid": "88ca8ab4-abd4-11e8-bb03-9a0015cd93e0"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>


## Two Continuous Variables-Plotted Two Ways


```python
xi = df_transform.valence
y = df_transform.danceability

# Generated linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
line = slope*xi+intercept

# Creating the dataset, and generating the plot
trace1 = go.Scatter(
                  x=xi,
                  y=y,
                  text = df_transform.name,
                  mode='markers',
                  marker=go.Marker(color='rgb(255, 127, 14)'),
                  name='Data'
                  )

trace2 = go.Scatter(
                  x=xi,
                  y=line,
                  mode='lines',
                  marker=go.Marker(color='rgb(31, 119, 180)'),
                  name='Fit'
                  )

annotation = go.Annotation(
                  x=10,
                  y=80000,
                  text='$R^2 = 0.9551,\\Y = 0.716X + 19.18$',
                  showarrow=False,
                  font=go.Font(size=16)
                  )
data = [trace1, trace2]

layout = go.Layout(
    title='Danceability and Valence',
    xaxis=dict(
        title='Valence',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Danceability',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)


fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Linear-Fit-in-python')
print('Slope:')
print(slope)
```


<div id="b10598e6-18d8-4936-9d9f-02a3dcf675b7" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
        Plotly.plot(
            'b10598e6-18d8-4936-9d9f-02a3dcf675b7',
            [{"marker": {"color": "rgb(255, 127, 14)"}, "mode": "markers", "name": "Data", "text": ["Shape of You", "Despacito - Remix", "Despacito (Featuring Daddy Yankee)", "Something Just Like This", "I'm the One", "HUMBLE.", "It Ain't Me (with Selena Gomez)", "Unforgettable", "That's What I Like", "I Don\u2019t Wanna Live Forever (Fifty Shades Darker) - From \"Fifty Shades Darker (Original Motion Picture Soundtrack)\"", "XO TOUR Llif3", "Paris", "Stay (with Alessia Cara)", "Attention", "Mask Off", "Congratulations", "Swalla (feat. Nicki Minaj & Ty Dolla $ign)", "Castle on the Hill", "Rockabye (feat. Sean Paul & Anne-Marie)", "Believer", "Mi Gente", "Thunder", "Say You Won't Let Go", "There's Nothing Holdin' Me Back", "Me Reh\u00faso", "Issues", "Galway Girl", "Scared to Be Lonely", "Closer", "Symphony (feat. Zara Larsson)", "I Feel It Coming", "Starboy", "Wild Thoughts", "Slide", "New Rules", "1-800-273-8255", "Passionfruit", "rockstar", "Strip That Down", "2U (feat. Justin Bieber)", "Perfect", "Call On Me - Ryan Riback Extended Remix", "Feels", "Mama", "Felices los 4", "iSpy (feat. Lil Yachty)", "Location", "Chantaje", "Bad and Boujee (feat. Lil Uzi Vert)", "Havana", "Solo Dance", "Fake Love", "Let Me Love You", "More Than You Know", "One Dance", "SUBEME LA RADIO", "Pretty Girl - Cheat Codes X CADE Remix", "Sorry Not Sorry", "Redbone", "24K Magic", "DNA.", "El Amante", "You Don't Know Me - Radio Edit", "Chained To The Rhythm", "No Promises (feat. Demi Lovato)", "Don't Wanna Know (feat. Kendrick Lamar)", "How Far I'll Go - From \"Moana\"", "Slow Hands", "Esc\u00e1pate Conmigo", "Bounce Back", "Sign of the Times", "goosebumps", "Young Dumb & Broke", "There for You", "Cold (feat. Future)", "Silence", "Too Good At Goodbyes", "Just Hold On", "Look What You Made Me Do", "Glorious (feat. Skylar Grey)", "Starving", "Reggaet\u00f3n Lento (Bailemos)", "Weak", "Side To Side", "Otra Vez (feat. J Balvin)", "I Like Me Better", "In the Name of Love", "Cold Water (feat. Justin Bieber & M\u00d8)", "Malibu", "All Night", "Hear Me Now", "Your Song", "Ahora Dice", "Friends (with BloodPop\u00ae)", "Bank Account", "Bad Things (with Camila Cabello)", "Don't Let Me Down", "Body Like A Back Road", "Now Or Never", "Dusk Till Dawn - Radio Edit"], "x": [0.9309999999999999, 0.813, 0.846, 0.446, 0.8109999999999999, 0.4, 0.515, 0.733, 0.86, 0.0862, 0.401, 0.21899999999999997, 0.498, 0.777, 0.281, 0.504, 0.782, 0.47100000000000003, 0.742, 0.708, 0.294, 0.298, 0.494, 0.966, 0.426, 0.42, 0.7809999999999999, 0.195, 0.6609999999999999, 0.457, 0.579, 0.535, 0.632, 0.511, 0.6559999999999999, 0.386, 0.364, 0.11900000000000001, 0.527, 0.557, 0.168, 0.718, 0.872, 0.557, 0.737, 0.672, 0.326, 0.907, 0.175, 0.418, 0.36, 0.605, 0.142, 0.544, 0.371, 0.647, 0.733, 0.863, 0.5870000000000001, 0.632, 0.402, 0.732, 0.682, 0.462, 0.595, 0.485, 0.159, 0.868, 0.754, 0.26, 0.222, 0.43, 0.39399999999999996, 0.13, 0.506, 0.28600000000000003, 0.534, 0.374, 0.49700000000000005, 0.35600000000000004, 0.5579999999999999, 0.71, 0.6779999999999999, 0.603, 0.7040000000000001, 0.419, 0.196, 0.501, 0.34299999999999997, 0.44799999999999995, 0.45, 0.9620000000000001, 0.42700000000000005, 0.649, 0.376, 0.272, 0.40299999999999997, 0.631, 0.434, 0.0967], "y": [0.825, 0.6940000000000001, 0.66, 0.617, 0.609, 0.904, 0.64, 0.726, 0.853, 0.735, 0.732, 0.653, 0.679, 0.774, 0.833, 0.627, 0.696, 0.461, 0.72, 0.779, 0.5429999999999999, 0.6, 0.358, 0.857, 0.7440000000000001, 0.706, 0.624, 0.584, 0.748, 0.7070000000000001, 0.768, 0.6809999999999999, 0.6709999999999999, 0.736, 0.7709999999999999, 0.629, 0.809, 0.5770000000000001, 0.8690000000000001, 0.5479999999999999, 0.599, 0.6759999999999999, 0.893, 0.746, 0.755, 0.746, 0.736, 0.852, 0.927, 0.768, 0.7440000000000001, 0.927, 0.47600000000000003, 0.644, 0.7909999999999999, 0.684, 0.703, 0.7040000000000001, 0.743, 0.818, 0.637, 0.6829999999999999, 0.8759999999999999, 0.44799999999999995, 0.741, 0.775, 0.314, 0.7340000000000001, 0.747, 0.77, 0.516, 0.841, 0.7979999999999999, 0.611, 0.6970000000000001, 0.52, 0.698, 0.647, 0.773, 0.731, 0.721, 0.7609999999999999, 0.6729999999999999, 0.648, 0.8320000000000001, 0.752, 0.49, 0.608, 0.573, 0.544, 0.789, 0.855, 0.708, 0.7440000000000001, 0.884, 0.675, 0.542, 0.731, 0.6579999999999999, 0.258], "type": "scatter", "uid": "8a8286c6-aa4f-11e8-ba63-9a0015cd93e0"}, {"marker": {"color": "rgb(31, 119, 180)"}, "mode": "lines", "name": "Fit", "x": [0.9309999999999999, 0.813, 0.846, 0.446, 0.8109999999999999, 0.4, 0.515, 0.733, 0.86, 0.0862, 0.401, 0.21899999999999997, 0.498, 0.777, 0.281, 0.504, 0.782, 0.47100000000000003, 0.742, 0.708, 0.294, 0.298, 0.494, 0.966, 0.426, 0.42, 0.7809999999999999, 0.195, 0.6609999999999999, 0.457, 0.579, 0.535, 0.632, 0.511, 0.6559999999999999, 0.386, 0.364, 0.11900000000000001, 0.527, 0.557, 0.168, 0.718, 0.872, 0.557, 0.737, 0.672, 0.326, 0.907, 0.175, 0.418, 0.36, 0.605, 0.142, 0.544, 0.371, 0.647, 0.733, 0.863, 0.5870000000000001, 0.632, 0.402, 0.732, 0.682, 0.462, 0.595, 0.485, 0.159, 0.868, 0.754, 0.26, 0.222, 0.43, 0.39399999999999996, 0.13, 0.506, 0.28600000000000003, 0.534, 0.374, 0.49700000000000005, 0.35600000000000004, 0.5579999999999999, 0.71, 0.6779999999999999, 0.603, 0.7040000000000001, 0.419, 0.196, 0.501, 0.34299999999999997, 0.44799999999999995, 0.45, 0.9620000000000001, 0.42700000000000005, 0.649, 0.376, 0.272, 0.40299999999999997, 0.631, 0.434, 0.0967], "y": [0.7981160774021486, 0.7692408309757515, 0.7773161117560151, 0.6794339204800923, 0.7687514200193719, 0.6681774684833612, 0.6963185984751891, 0.749664392720567, 0.7807419884506723, 0.5913888894273999, 0.6684221739615511, 0.6238857769310062, 0.6921586053459623, 0.7604314337609184, 0.6390575165787743, 0.6936268382151012, 0.7616549611518675, 0.6855515574348375, 0.7518667420242752, 0.7435467557658217, 0.6422386877952417, 0.643217509708001, 0.6911797834332031, 0.8066807691387919, 0.6745398109162962, 0.6730715780471574, 0.7614102556736776, 0.6180128454544509, 0.7320455982909009, 0.6821256807401803, 0.7119797490793367, 0.7012127080389852, 0.7249491394233964, 0.6953397765624298, 0.7308220708999518, 0.664751591788704, 0.6593680712685283, 0.5994152291120256, 0.6992550642134667, 0.7065962285591609, 0.6114057975433261, 0.7459938105477198, 0.7836784541889501, 0.7065962285591609, 0.7506432146333262, 0.7347373585509888, 0.6500692630973156, 0.7922431459255933, 0.6131187358906547, 0.6725821670907778, 0.658389249355769, 0.7183420915122717, 0.6050434551103911, 0.7034150573426935, 0.6610810096158569, 0.7286197215962436, 0.749664392720567, 0.7814761048852418, 0.7139373929048551, 0.7249491394233964, 0.6686668794397409, 0.7494196872423771, 0.7371844133328868, 0.6833492081311293, 0.7158950367303736, 0.6889774341294949, 0.6092034482396178, 0.7826996322761909, 0.7548032077625528, 0.6339187015367883, 0.6246198933655757, 0.6755186328290554, 0.6667092356142225, 0.6021069893721135, 0.6941162491714807, 0.6402810439697233, 0.7009680025607954, 0.6618151260504263, 0.6919138998677725, 0.6574104274430098, 0.7068409340373507, 0.7440361667222013, 0.7362055914201275, 0.717852680555892, 0.7425679338530625, 0.6728268725689676, 0.6182575509326407, 0.6928927217805317, 0.6542292562265423, 0.679923331436472, 0.6804127423928517, 0.8057019472260327, 0.6747845163944861, 0.7291091325526231, 0.6623045370068059, 0.636855167275066, 0.6689115849179307, 0.7247044339452067, 0.6764974547418147, 0.5939582969483929], "type": "scatter", "uid": "8a828874-aa4f-11e8-bdba-9a0015cd93e0"}],
            {"title": "Danceability and Valence", "xaxis": {"title": "Valence", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Danceability", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}},
            {"showLink": true, "linkText": "Export to plot.ly"}
        ).then(function () {return Plotly.addFrames('b10598e6-18d8-4936-9d9f-02a3dcf675b7',{});}).then(function(){Plotly.animate('b10598e6-18d8-4936-9d9f-02a3dcf675b7');})
        });</script>


    Slope:
    0.24470547818980679



```python
x = df_transform.valence
y = df_transform.danceability

trace = [go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Jet',
        contours = dict(
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                color = 'white'
            )
        ),
        hoverlabel = dict(
            bgcolor = 'white',
            bordercolor = 'black',
            font = dict(
                family = 'Raleway',
                color = 'black'
            )
        )
        
)]

data = trace

layout = go.Layout(
    title='Danceability and Valence',
    xaxis=dict(
        title='Valence',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Danceability',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
```


<div id="5c922324-6db2-4776-ba7e-b503709ada5a" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
        Plotly.plot(
            '5c922324-6db2-4776-ba7e-b503709ada5a',
            [{"colorscale": "Jet", "contours": {"labelfont": {"color": "white", "family": "Raleway"}, "showlabels": true}, "hoverlabel": {"bgcolor": "white", "bordercolor": "black", "font": {"color": "black", "family": "Raleway"}}, "x": [0.9309999999999999, 0.813, 0.846, 0.446, 0.8109999999999999, 0.4, 0.515, 0.733, 0.86, 0.0862, 0.401, 0.21899999999999997, 0.498, 0.777, 0.281, 0.504, 0.782, 0.47100000000000003, 0.742, 0.708, 0.294, 0.298, 0.494, 0.966, 0.426, 0.42, 0.7809999999999999, 0.195, 0.6609999999999999, 0.457, 0.579, 0.535, 0.632, 0.511, 0.6559999999999999, 0.386, 0.364, 0.11900000000000001, 0.527, 0.557, 0.168, 0.718, 0.872, 0.557, 0.737, 0.672, 0.326, 0.907, 0.175, 0.418, 0.36, 0.605, 0.142, 0.544, 0.371, 0.647, 0.733, 0.863, 0.5870000000000001, 0.632, 0.402, 0.732, 0.682, 0.462, 0.595, 0.485, 0.159, 0.868, 0.754, 0.26, 0.222, 0.43, 0.39399999999999996, 0.13, 0.506, 0.28600000000000003, 0.534, 0.374, 0.49700000000000005, 0.35600000000000004, 0.5579999999999999, 0.71, 0.6779999999999999, 0.603, 0.7040000000000001, 0.419, 0.196, 0.501, 0.34299999999999997, 0.44799999999999995, 0.45, 0.9620000000000001, 0.42700000000000005, 0.649, 0.376, 0.272, 0.40299999999999997, 0.631, 0.434, 0.0967], "y": [0.825, 0.6940000000000001, 0.66, 0.617, 0.609, 0.904, 0.64, 0.726, 0.853, 0.735, 0.732, 0.653, 0.679, 0.774, 0.833, 0.627, 0.696, 0.461, 0.72, 0.779, 0.5429999999999999, 0.6, 0.358, 0.857, 0.7440000000000001, 0.706, 0.624, 0.584, 0.748, 0.7070000000000001, 0.768, 0.6809999999999999, 0.6709999999999999, 0.736, 0.7709999999999999, 0.629, 0.809, 0.5770000000000001, 0.8690000000000001, 0.5479999999999999, 0.599, 0.6759999999999999, 0.893, 0.746, 0.755, 0.746, 0.736, 0.852, 0.927, 0.768, 0.7440000000000001, 0.927, 0.47600000000000003, 0.644, 0.7909999999999999, 0.684, 0.703, 0.7040000000000001, 0.743, 0.818, 0.637, 0.6829999999999999, 0.8759999999999999, 0.44799999999999995, 0.741, 0.775, 0.314, 0.7340000000000001, 0.747, 0.77, 0.516, 0.841, 0.7979999999999999, 0.611, 0.6970000000000001, 0.52, 0.698, 0.647, 0.773, 0.731, 0.721, 0.7609999999999999, 0.6729999999999999, 0.648, 0.8320000000000001, 0.752, 0.49, 0.608, 0.573, 0.544, 0.789, 0.855, 0.708, 0.7440000000000001, 0.884, 0.675, 0.542, 0.731, 0.6579999999999999, 0.258], "type": "histogram2dcontour", "uid": "1d9ae266-a9e3-11e8-b1c6-9a0015cd93e0"}],
            {"title": "Danceability and Valence", "xaxis": {"title": "Valence", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Danceability", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}},
            {"showLink": true, "linkText": "Export to plot.ly"}
        ).then(function () {return Plotly.addFrames('5c922324-6db2-4776-ba7e-b503709ada5a',{});}).then(function(){Plotly.animate('5c922324-6db2-4776-ba7e-b503709ada5a');})
        });</script>


## BPM (Tempo) and Key- 6 Different Ways


```python
x = df_transform.tempo
data = [go.Histogram(x=x)]
layout = go.Layout(
    title='Frequency of BPMs',
    xaxis=dict(
        title='Beats Per Minute',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Number of Songs',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='basic histogram')
```


<div id="d7e8fc65-82c2-4139-a739-d048aa744332" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("d7e8fc65-82c2-4139-a739-d048aa744332", [{"x": [95.977, 88.931, 177.833, 103.01899999999999, 80.92399999999999, 150.02, 99.96799999999999, 97.985, 134.066, 117.973, 155.096, 99.99, 102.01299999999999, 100.041, 150.062, 123.071, 98.064, 135.007, 101.965, 124.98200000000001, 103.809, 167.88, 85.04299999999999, 121.99600000000001, 104.823, 113.804, 99.943, 137.972, 95.01, 122.863, 92.994, 186.054, 97.98, 104.066, 116.054, 100.015, 111.98, 159.77200000000002, 106.02799999999999, 144.937, 95.05, 105.00299999999999, 101.01799999999999, 104.027, 93.973, 75.016, 80.126, 102.03399999999999, 127.07600000000001, 104.992, 114.965, 133.987, 199.864, 123.074, 103.98899999999999, 91.04799999999999, 121.03, 144.02100000000002, 160.083, 106.97, 139.931, 179.91, 124.007, 189.798, 112.956, 100.04799999999999, 179.666, 85.90899999999999, 92.02799999999999, 81.477, 119.97200000000001, 130.049, 136.94899999999998, 105.969, 99.905, 141.971, 91.92, 114.991, 128.062, 139.994, 99.914, 93.97399999999999, 123.98, 159.145, 96.016, 91.97, 133.889, 92.943, 139.934, 145.017, 121.971, 117.959, 143.965, 104.99, 75.016, 137.817, 159.797, 98.963, 110.075, 180.043], "type": "histogram", "uid": "ae74adfa-aa4f-11e8-98ad-9a0015cd93e0"}], {"title": "Frequency of BPMs", "xaxis": {"title": "Beats Per Minute", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Number of Songs", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
x = df_transform.key_change
y = df_transform.tempo

data = [dict(
  type = 'scatter',
      x = x,
      y = y,
      mode = 'markers',
      text = df_transform.name,
      transforms = [dict(
        type = 'groupby',
        groups = key,
        styles = [
            dict(target = 'C', value = dict(marker = dict(color = 'rgb(244, 92, 66)'))),
            dict(target = 'C#/Db', value = dict(marker = dict(color = 'rgb(244, 178, 65)'))),
            dict(target = 'D', value = dict(marker = dict(color = 'rgb(244, 244, 65)'))),
            dict(target = 'Eb', value = dict(marker = dict(color = 'rgb(133, 244, 65)'))),
            dict(target = 'E', value = dict(marker = dict(color = 'rgb(17, 104, 46)'))),
            dict(target = 'F', value = dict(marker = dict(color = 'rgb(15, 226, 202)'))),
            dict(target = 'F#/Gb', value = dict(marker = dict(color = 'rgb(14, 81, 226)'))),
            dict(target = 'G', value = dict(marker = dict(color = 'rgb(138, 100, 209)'))),
            dict(target = 'Ab', value = dict(marker = dict(color = 'rgb(219, 15, 226)'))),
            dict(target = 'A', value = dict(marker = dict(color = 'rgb(155, 10, 68)'))),
            dict(target = 'Bb', value = dict(marker = dict(color = 'rgb(130, 126, 127)'))),
            dict(target = 'B', value = dict(marker = dict(color = 'rgb(117, 42, 42)')))
        ]
  )]
)]

layout = dict(
    title='Tempo by Key of Song',
    xaxis=dict(
        title='Key of Song',
        titlefont=dict(
            family='Raleway, sans-serif',
            size=18,
            color='black'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='all'
        
    ),
    yaxis=dict(
        title='Tempo/Beats Per Minute',
        titlefont=dict(
            family='Raleway, sans-serif',
            size=18,
            color='black'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Raleway, sans-serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='all'
    )
)

iplot({'data': data, 'layout': layout}, validate=False)

#ticktext=['January', 'February', ...]

```


<div id="a2557018-58df-4b5b-94e4-9e5c2b6f587f" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("a2557018-58df-4b5b-94e4-9e5c2b6f587f", [{"type": "scatter", "x": ["C#/Db", "D", "D", "B", "G", "C#/Db", "C", "F#/Gb", "C#/Db", "C", "B", "D", "F", "Eb", "D", "F#/Gb", "C#/Db", "D", "A", "Bb", "B", "C", "Bb", "D", "C#/Db", "Ab", "A", "C#/Db", "Ab", "C", "C", "G", "C", "C#/Db", "A", "F", "B", "F", "F#/Gb", "Ab", "Ab", "C", "B", "B", "F", "G", "C#/Db", "Ab", "B", "G", "F#/Gb", "A", "Ab", "F", "C#/Db", "A", "G", "B", "C#/Db", "C#/Db", "C#/Db", "Ab", "B", "C", "Bb", "G", "A", "C", "Ab", "D", "F", "G", "C#/Db", "F#/Gb", "A", "E", "F", "B", "A", "C", "E", "E", "F", "F#/Gb", "Bb", "A", "E", "F#/Gb", "Ab", "Ab", "B", "C#/Db", "F#/Gb", "Ab", "Ab", "D", "B", "F", "F#/Gb", "B"], "y": [95.977, 88.931, 177.833, 103.01899999999999, 80.92399999999999, 150.02, 99.96799999999999, 97.985, 134.066, 117.973, 155.096, 99.99, 102.01299999999999, 100.041, 150.062, 123.071, 98.064, 135.007, 101.965, 124.98200000000001, 103.809, 167.88, 85.04299999999999, 121.99600000000001, 104.823, 113.804, 99.943, 137.972, 95.01, 122.863, 92.994, 186.054, 97.98, 104.066, 116.054, 100.015, 111.98, 159.77200000000002, 106.02799999999999, 144.937, 95.05, 105.00299999999999, 101.01799999999999, 104.027, 93.973, 75.016, 80.126, 102.03399999999999, 127.07600000000001, 104.992, 114.965, 133.987, 199.864, 123.074, 103.98899999999999, 91.04799999999999, 121.03, 144.02100000000002, 160.083, 106.97, 139.931, 179.91, 124.007, 189.798, 112.956, 100.04799999999999, 179.666, 85.90899999999999, 92.02799999999999, 81.477, 119.97200000000001, 130.049, 136.94899999999998, 105.969, 99.905, 141.971, 91.92, 114.991, 128.062, 139.994, 99.914, 93.97399999999999, 123.98, 159.145, 96.016, 91.97, 133.889, 92.943, 139.934, 145.017, 121.971, 117.959, 143.965, 104.99, 75.016, 137.817, 159.797, 98.963, 110.075, 180.043], "mode": "markers", "text": ["Shape of You", "Despacito - Remix", "Despacito (Featuring Daddy Yankee)", "Something Just Like This", "I'm the One", "HUMBLE.", "It Ain't Me (with Selena Gomez)", "Unforgettable", "That's What I Like", "I Don\u2019t Wanna Live Forever (Fifty Shades Darker) - From \"Fifty Shades Darker (Original Motion Picture Soundtrack)\"", "XO TOUR Llif3", "Paris", "Stay (with Alessia Cara)", "Attention", "Mask Off", "Congratulations", "Swalla (feat. Nicki Minaj & Ty Dolla $ign)", "Castle on the Hill", "Rockabye (feat. Sean Paul & Anne-Marie)", "Believer", "Mi Gente", "Thunder", "Say You Won't Let Go", "There's Nothing Holdin' Me Back", "Me Reh\u00faso", "Issues", "Galway Girl", "Scared to Be Lonely", "Closer", "Symphony (feat. Zara Larsson)", "I Feel It Coming", "Starboy", "Wild Thoughts", "Slide", "New Rules", "1-800-273-8255", "Passionfruit", "rockstar", "Strip That Down", "2U (feat. Justin Bieber)", "Perfect", "Call On Me - Ryan Riback Extended Remix", "Feels", "Mama", "Felices los 4", "iSpy (feat. Lil Yachty)", "Location", "Chantaje", "Bad and Boujee (feat. Lil Uzi Vert)", "Havana", "Solo Dance", "Fake Love", "Let Me Love You", "More Than You Know", "One Dance", "SUBEME LA RADIO", "Pretty Girl - Cheat Codes X CADE Remix", "Sorry Not Sorry", "Redbone", "24K Magic", "DNA.", "El Amante", "You Don't Know Me - Radio Edit", "Chained To The Rhythm", "No Promises (feat. Demi Lovato)", "Don't Wanna Know (feat. Kendrick Lamar)", "How Far I'll Go - From \"Moana\"", "Slow Hands", "Esc\u00e1pate Conmigo", "Bounce Back", "Sign of the Times", "goosebumps", "Young Dumb & Broke", "There for You", "Cold (feat. Future)", "Silence", "Too Good At Goodbyes", "Just Hold On", "Look What You Made Me Do", "Glorious (feat. Skylar Grey)", "Starving", "Reggaet\u00f3n Lento (Bailemos)", "Weak", "Side To Side", "Otra Vez (feat. J Balvin)", "I Like Me Better", "In the Name of Love", "Cold Water (feat. Justin Bieber & M\u00d8)", "Malibu", "All Night", "Hear Me Now", "Your Song", "Ahora Dice", "Friends (with BloodPop\u00ae)", "Bank Account", "Bad Things (with Camila Cabello)", "Don't Let Me Down", "Body Like A Back Road", "Now Or Never", "Dusk Till Dawn - Radio Edit"], "transforms": [{"type": "groupby", "groups": [1.0, 2.0, 2.0, 11.0, 7.0, 1.0, 0.0, 6.0, 1.0, 0.0, 11.0, 2.0, 5.0, 3.0, 2.0, 6.0, 1.0, 2.0, 9.0, 10.0, 11.0, 0.0, 10.0, 2.0, 1.0, 8.0, 9.0, 1.0, 8.0, 0.0, 0.0, 7.0, 0.0, 1.0, 9.0, 5.0, 11.0, 5.0, 6.0, 8.0, 8.0, 0.0, 11.0, 11.0, 5.0, 7.0, 1.0, 8.0, 11.0, 7.0, 6.0, 9.0, 8.0, 5.0, 1.0, 9.0, 7.0, 11.0, 1.0, 1.0, 1.0, 8.0, 11.0, 0.0, 10.0, 7.0, 9.0, 0.0, 8.0, 2.0, 5.0, 7.0, 1.0, 6.0, 9.0, 4.0, 5.0, 11.0, 9.0, 0.0, 4.0, 4.0, 5.0, 6.0, 10.0, 9.0, 4.0, 6.0, 8.0, 8.0, 11.0, 1.0, 6.0, 8.0, 8.0, 2.0, 11.0, 5.0, 6.0, 11.0], "styles": [{"target": "C", "value": {"marker": {"color": "rgb(244, 92, 66)"}}}, {"target": "C#/Db", "value": {"marker": {"color": "rgb(244, 178, 65)"}}}, {"target": "D", "value": {"marker": {"color": "rgb(244, 244, 65)"}}}, {"target": "Eb", "value": {"marker": {"color": "rgb(133, 244, 65)"}}}, {"target": "E", "value": {"marker": {"color": "rgb(17, 104, 46)"}}}, {"target": "F", "value": {"marker": {"color": "rgb(15, 226, 202)"}}}, {"target": "F#/Gb", "value": {"marker": {"color": "rgb(14, 81, 226)"}}}, {"target": "G", "value": {"marker": {"color": "rgb(138, 100, 209)"}}}, {"target": "Ab", "value": {"marker": {"color": "rgb(219, 15, 226)"}}}, {"target": "A", "value": {"marker": {"color": "rgb(155, 10, 68)"}}}, {"target": "Bb", "value": {"marker": {"color": "rgb(130, 126, 127)"}}}, {"target": "B", "value": {"marker": {"color": "rgb(117, 42, 42)"}}}]}]}], {"title": "Tempo by Key of Song", "xaxis": {"title": "Key of Song", "titlefont": {"family": "Raleway, sans-serif", "size": 18, "color": "black"}, "showticklabels": true, "tickangle": 45, "tickfont": {"family": "Old Standard TT, serif", "size": 14, "color": "black"}, "exponentformat": "e", "showexponent": "all"}, "yaxis": {"title": "Tempo/Beats Per Minute", "titlefont": {"family": "Raleway, sans-serif", "size": 18, "color": "black"}, "showticklabels": true, "tickangle": 45, "tickfont": {"family": "Raleway, sans-serif", "size": 14, "color": "black"}, "exponentformat": "e", "showexponent": "all"}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
x = df_transform.key
y = df_transform.tempo

data = [
    go.Histogram2d(x=x, y=y, histnorm='probability',
        autobinx=False,
        xbins=dict(start=0, end=11, size=1),
        autobiny=False,
        ybins=dict(start=70, end=200, size=10),
        colorscale='Portland'
    )
]

layout = dict(
    title='Tempo by Key of Song',
    xaxis=dict(
        title='Key of Song',
        titlefont=dict(
            family='Raleway, sans-serif',
            size=18,
            color='black'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Old Standard TT, serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='all'
    ),
    yaxis=dict(
        title='Tempo/Beats Per Minute',
        titlefont=dict(
            family='Raleway, sans-serif',
            size=16,
            color='black'
        ),
        showticklabels=True,
        tickangle=45,
        tickfont=dict(
            family='Raleway, sans-serif',
            size=14,
            color='black'
        ),
        exponentformat='e',
        showexponent='all'
    )
)

iplot({'data': data, 'layout': layout}, validate=False)
```


<div id="f1d78c86-a851-45d7-933c-2c8d5145401c" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("f1d78c86-a851-45d7-933c-2c8d5145401c", [{"autobinx": false, "autobiny": false, "colorscale": "Portland", "histnorm": "probability", "x": [1.0, 2.0, 2.0, 11.0, 7.0, 1.0, 0.0, 6.0, 1.0, 0.0, 11.0, 2.0, 5.0, 3.0, 2.0, 6.0, 1.0, 2.0, 9.0, 10.0, 11.0, 0.0, 10.0, 2.0, 1.0, 8.0, 9.0, 1.0, 8.0, 0.0, 0.0, 7.0, 0.0, 1.0, 9.0, 5.0, 11.0, 5.0, 6.0, 8.0, 8.0, 0.0, 11.0, 11.0, 5.0, 7.0, 1.0, 8.0, 11.0, 7.0, 6.0, 9.0, 8.0, 5.0, 1.0, 9.0, 7.0, 11.0, 1.0, 1.0, 1.0, 8.0, 11.0, 0.0, 10.0, 7.0, 9.0, 0.0, 8.0, 2.0, 5.0, 7.0, 1.0, 6.0, 9.0, 4.0, 5.0, 11.0, 9.0, 0.0, 4.0, 4.0, 5.0, 6.0, 10.0, 9.0, 4.0, 6.0, 8.0, 8.0, 11.0, 1.0, 6.0, 8.0, 8.0, 2.0, 11.0, 5.0, 6.0, 11.0], "xbins": {"end": 11, "size": 1, "start": 0}, "y": [95.977, 88.931, 177.833, 103.01899999999999, 80.92399999999999, 150.02, 99.96799999999999, 97.985, 134.066, 117.973, 155.096, 99.99, 102.01299999999999, 100.041, 150.062, 123.071, 98.064, 135.007, 101.965, 124.98200000000001, 103.809, 167.88, 85.04299999999999, 121.99600000000001, 104.823, 113.804, 99.943, 137.972, 95.01, 122.863, 92.994, 186.054, 97.98, 104.066, 116.054, 100.015, 111.98, 159.77200000000002, 106.02799999999999, 144.937, 95.05, 105.00299999999999, 101.01799999999999, 104.027, 93.973, 75.016, 80.126, 102.03399999999999, 127.07600000000001, 104.992, 114.965, 133.987, 199.864, 123.074, 103.98899999999999, 91.04799999999999, 121.03, 144.02100000000002, 160.083, 106.97, 139.931, 179.91, 124.007, 189.798, 112.956, 100.04799999999999, 179.666, 85.90899999999999, 92.02799999999999, 81.477, 119.97200000000001, 130.049, 136.94899999999998, 105.969, 99.905, 141.971, 91.92, 114.991, 128.062, 139.994, 99.914, 93.97399999999999, 123.98, 159.145, 96.016, 91.97, 133.889, 92.943, 139.934, 145.017, 121.971, 117.959, 143.965, 104.99, 75.016, 137.817, 159.797, 98.963, 110.075, 180.043], "ybins": {"end": 200, "size": 10, "start": 70}, "type": "histogram2d"}], {"title": "Tempo by Key of Song", "xaxis": {"title": "Key of Song", "titlefont": {"family": "Raleway, sans-serif", "size": 18, "color": "black"}, "showticklabels": true, "tickangle": 45, "tickfont": {"family": "Old Standard TT, serif", "size": 14, "color": "black"}, "exponentformat": "e", "showexponent": "all"}, "yaxis": {"title": "Tempo/Beats Per Minute", "titlefont": {"family": "Raleway, sans-serif", "size": 16, "color": "black"}, "showticklabels": true, "tickangle": 45, "tickfont": {"family": "Raleway, sans-serif", "size": 14, "color": "black"}, "exponentformat": "e", "showexponent": "all"}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
x = df_transform.key_change
y = df_transform.tempo

trace = [go.Histogram2dContour(
        x = x,
        y = y,
        colorscale = 'Viridis',
        contours = dict(
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                color = 'white'
            )
        ),
        hoverlabel = dict(
            bgcolor = 'white',
            bordercolor = 'black',
            font = dict(
                family = 'Raleway',
                color = 'black'
            )
        )
        
)]

data = trace

layout = go.Layout(
    title='Key and Tempo',
    xaxis=dict(
        title='Key',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Tempo',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
```


<div id="a8009280-0e34-4faf-b013-accc7f0be8d1" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
        Plotly.plot(
            'a8009280-0e34-4faf-b013-accc7f0be8d1',
            [{"colorscale": "Viridis", "contours": {"labelfont": {"color": "white", "family": "Raleway"}, "showlabels": true}, "hoverlabel": {"bgcolor": "white", "bordercolor": "black", "font": {"color": "black", "family": "Raleway"}}, "x": ["C#/Db", "D", "D", "B", "G", "C#/Db", "C", "F#/Gb", "C#/Db", "C", "B", "D", "F", "Eb", "D", "F#/Gb", "C#/Db", "D", "A", "Bb", "B", "C", "Bb", "D", "C#/Db", "Ab", "A", "C#/Db", "Ab", "C", "C", "G", "C", "C#/Db", "A", "F", "B", "F", "F#/Gb", "Ab", "Ab", "C", "B", "B", "F", "G", "C#/Db", "Ab", "B", "G", "F#/Gb", "A", "Ab", "F", "C#/Db", "A", "G", "B", "C#/Db", "C#/Db", "C#/Db", "Ab", "B", "C", "Bb", "G", "A", "C", "Ab", "D", "F", "G", "C#/Db", "F#/Gb", "A", "E", "F", "B", "A", "C", "E", "E", "F", "F#/Gb", "Bb", "A", "E", "F#/Gb", "Ab", "Ab", "B", "C#/Db", "F#/Gb", "Ab", "Ab", "D", "B", "F", "F#/Gb", "B"], "y": [95.977, 88.931, 177.833, 103.01899999999999, 80.92399999999999, 150.02, 99.96799999999999, 97.985, 134.066, 117.973, 155.096, 99.99, 102.01299999999999, 100.041, 150.062, 123.071, 98.064, 135.007, 101.965, 124.98200000000001, 103.809, 167.88, 85.04299999999999, 121.99600000000001, 104.823, 113.804, 99.943, 137.972, 95.01, 122.863, 92.994, 186.054, 97.98, 104.066, 116.054, 100.015, 111.98, 159.77200000000002, 106.02799999999999, 144.937, 95.05, 105.00299999999999, 101.01799999999999, 104.027, 93.973, 75.016, 80.126, 102.03399999999999, 127.07600000000001, 104.992, 114.965, 133.987, 199.864, 123.074, 103.98899999999999, 91.04799999999999, 121.03, 144.02100000000002, 160.083, 106.97, 139.931, 179.91, 124.007, 189.798, 112.956, 100.04799999999999, 179.666, 85.90899999999999, 92.02799999999999, 81.477, 119.97200000000001, 130.049, 136.94899999999998, 105.969, 99.905, 141.971, 91.92, 114.991, 128.062, 139.994, 99.914, 93.97399999999999, 123.98, 159.145, 96.016, 91.97, 133.889, 92.943, 139.934, 145.017, 121.971, 117.959, 143.965, 104.99, 75.016, 137.817, 159.797, 98.963, 110.075, 180.043], "type": "histogram2dcontour", "uid": "9e150328-abd9-11e8-adaa-9a0015cd93e0"}],
            {"title": "Key and Tempo", "xaxis": {"title": "Key", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Tempo", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}},
            {"showLink": true, "linkText": "Export to plot.ly"}
        ).then(function () {return Plotly.addFrames('a8009280-0e34-4faf-b013-accc7f0be8d1',{});}).then(function(){Plotly.animate('a8009280-0e34-4faf-b013-accc7f0be8d1');})
        });</script>



```python
data = [go.Bar(x=df_transform.key_change,
            y=df_transform.tempo)]
layout = go.Layout(
    title='Energy by Song',
    xaxis=dict(
        title='Name',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Energy',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = dict(data=data, layout=layout)
iplot(fig, filename='basic-bar')
```


<div id="9437d393-750d-4ef3-944f-74db10b8c0cd" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("9437d393-750d-4ef3-944f-74db10b8c0cd", [{"x": ["C#/Db", "D", "D", "B", "G", "C#/Db", "C", "F#/Gb", "C#/Db", "C", "B", "D", "F", "Eb", "D", "F#/Gb", "C#/Db", "D", "A", "Bb", "B", "C", "Bb", "D", "C#/Db", "Ab", "A", "C#/Db", "Ab", "C", "C", "G", "C", "C#/Db", "A", "F", "B", "F", "F#/Gb", "Ab", "Ab", "C", "B", "B", "F", "G", "C#/Db", "Ab", "B", "G", "F#/Gb", "A", "Ab", "F", "C#/Db", "A", "G", "B", "C#/Db", "C#/Db", "C#/Db", "Ab", "B", "C", "Bb", "G", "A", "C", "Ab", "D", "F", "G", "C#/Db", "F#/Gb", "A", "E", "F", "B", "A", "C", "E", "E", "F", "F#/Gb", "Bb", "A", "E", "F#/Gb", "Ab", "Ab", "B", "C#/Db", "F#/Gb", "Ab", "Ab", "D", "B", "F", "F#/Gb", "B"], "y": [95.977, 88.931, 177.833, 103.01899999999999, 80.92399999999999, 150.02, 99.96799999999999, 97.985, 134.066, 117.973, 155.096, 99.99, 102.01299999999999, 100.041, 150.062, 123.071, 98.064, 135.007, 101.965, 124.98200000000001, 103.809, 167.88, 85.04299999999999, 121.99600000000001, 104.823, 113.804, 99.943, 137.972, 95.01, 122.863, 92.994, 186.054, 97.98, 104.066, 116.054, 100.015, 111.98, 159.77200000000002, 106.02799999999999, 144.937, 95.05, 105.00299999999999, 101.01799999999999, 104.027, 93.973, 75.016, 80.126, 102.03399999999999, 127.07600000000001, 104.992, 114.965, 133.987, 199.864, 123.074, 103.98899999999999, 91.04799999999999, 121.03, 144.02100000000002, 160.083, 106.97, 139.931, 179.91, 124.007, 189.798, 112.956, 100.04799999999999, 179.666, 85.90899999999999, 92.02799999999999, 81.477, 119.97200000000001, 130.049, 136.94899999999998, 105.969, 99.905, 141.971, 91.92, 114.991, 128.062, 139.994, 99.914, 93.97399999999999, 123.98, 159.145, 96.016, 91.97, 133.889, 92.943, 139.934, 145.017, 121.971, 117.959, 143.965, 104.99, 75.016, 137.817, 159.797, 98.963, 110.075, 180.043], "type": "bar", "uid": "e7b7b85e-abdb-11e8-99cd-9a0015cd93e0"}], {"title": "Energy by Song", "xaxis": {"title": "Name", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}, "yaxis": {"title": "Energyy", "titlefont": {"color": "#7f7f7f", "family": "Courier New, monospace", "size": 18}}}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>



```python
labels = df_transform.key_change
values = df_transform.tempo

trace = go.Pie(labels=labels, values=values)

iplot([trace], filename='basic_pie_chart')
```


<div id="c74636ff-e905-4a77-b906-66bb95990861" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";Plotly.newPlot("c74636ff-e905-4a77-b906-66bb95990861", [{"labels": ["C#/Db", "D", "D", "B", "G", "C#/Db", "C", "F#/Gb", "C#/Db", "C", "B", "D", "F", "Eb", "D", "F#/Gb", "C#/Db", "D", "A", "Bb", "B", "C", "Bb", "D", "C#/Db", "Ab", "A", "C#/Db", "Ab", "C", "C", "G", "C", "C#/Db", "A", "F", "B", "F", "F#/Gb", "Ab", "Ab", "C", "B", "B", "F", "G", "C#/Db", "Ab", "B", "G", "F#/Gb", "A", "Ab", "F", "C#/Db", "A", "G", "B", "C#/Db", "C#/Db", "C#/Db", "Ab", "B", "C", "Bb", "G", "A", "C", "Ab", "D", "F", "G", "C#/Db", "F#/Gb", "A", "E", "F", "B", "A", "C", "E", "E", "F", "F#/Gb", "Bb", "A", "E", "F#/Gb", "Ab", "Ab", "B", "C#/Db", "F#/Gb", "Ab", "Ab", "D", "B", "F", "F#/Gb", "B"], "values": [95.977, 88.931, 177.833, 103.01899999999999, 80.92399999999999, 150.02, 99.96799999999999, 97.985, 134.066, 117.973, 155.096, 99.99, 102.01299999999999, 100.041, 150.062, 123.071, 98.064, 135.007, 101.965, 124.98200000000001, 103.809, 167.88, 85.04299999999999, 121.99600000000001, 104.823, 113.804, 99.943, 137.972, 95.01, 122.863, 92.994, 186.054, 97.98, 104.066, 116.054, 100.015, 111.98, 159.77200000000002, 106.02799999999999, 144.937, 95.05, 105.00299999999999, 101.01799999999999, 104.027, 93.973, 75.016, 80.126, 102.03399999999999, 127.07600000000001, 104.992, 114.965, 133.987, 199.864, 123.074, 103.98899999999999, 91.04799999999999, 121.03, 144.02100000000002, 160.083, 106.97, 139.931, 179.91, 124.007, 189.798, 112.956, 100.04799999999999, 179.666, 85.90899999999999, 92.02799999999999, 81.477, 119.97200000000001, 130.049, 136.94899999999998, 105.969, 99.905, 141.971, 91.92, 114.991, 128.062, 139.994, 99.914, 93.97399999999999, 123.98, 159.145, 96.016, 91.97, 133.889, 92.943, 139.934, 145.017, 121.971, 117.959, 143.965, 104.99, 75.016, 137.817, 159.797, 98.963, 110.075, 180.043], "type": "pie", "uid": "1d81b4de-abdd-11e8-b4c0-9a0015cd93e0"}], {}, {"showLink": true, "linkText": "Export to plot.ly"})});</script>

