# From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs
Contextual automatic speech recognition (ASR) with Speech-LLMs is typically trained with oracle conversation history, but relies on error-prone history at inference, causing a train–test mismatch in the context channel that we term contextual exposure bias. We propose a unified training framework to improve robustness under realistic histories: (i) Teacher Error Knowledge by using Whisper large-v3 hypotheses as training-time history, (ii) Context Dropout to regularize over-reliance on history, and (iii) Direct Preference Optimization (DPO) on curated failure cases. Experiments on TED-LIUM 3 (in-domain) and zero-shot LibriSpeech (out-of-domain) show consistent gains under predicted-history decoding. With a two-utterance history, SFT with Whisper histories reduce WER from 5.59\% (oracle-history training) to 5.47\%, and DPO further improves to 5.17\%. Under irrelevant-history attacks, DPO yields the smallest degradation (5.17\% $\rightarrow$ 5.63\%), indicating improved robustness to misleading context.

<img src="fig/model.png" alt="model arc" width="60%">

## Installation

~~~
conda env create -f environment.yml
conda activate speechllm
~~~

## Checkpoints
The model checkpoints are available for anonymous review on Figshare and Zenodo: [Figshare](https://figshare.com/s/e8c84e55f36540c160b7), [Zenodo](https://zenodo.org/records/18181120?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6Ijk3N2YwOGZlLTEwODYtNDIyNS1iOTVlLWM4ZjViZTk3NDJkZSIsImRhdGEiOnt9LCJyYW5kb20iOiIxNGY2ZDU3Nzc2ZjA4ZGM3NGQ0MThjODk5NmQ4OWJkNSJ9.xhD-hOqHS-WAt-awXHMdWlPo_l3spRjUzfEo0sEPrO83o9xIsRvsZH1MD7k3Xpqh8ElJ6Y3-Brd0scKL_jv8bA)

## Results
### Main Result
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center; font-family: Arial, sans-serif; font-size: 14px;">
  <caption style="caption-side: bottom; text-align: left; padding-top: 10px; font-size: 13px; color: #000;">
    Table 1: WER comparison on TED-LIUM 3 and out-of-domain Librispeech dataset across different context window sizes (<i>N</i>). The column <b>Con<sub>inf</sub> / Con<sub>train</sub></b> specifies the source of history used during inference and training, respectively. <b>hyp</b> denotes using the model's own predictions as history during inference. Regarding training configuration, <b>GT</b> uses ground-truth history, while <b>Whisper</b> indicates the model was trained using context decoded by Whisper to simulate historical errors. <b>+ DPO</b> and <b>+ SFT2</b> are additional fine-tuning stages applied to the SFT model.
  </caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;"><i>N</i></th>
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">Con<sub>inf</sub>/Con<sub>train</sub></th>
      <th colspan="4" style="padding: 5px; border-bottom: 1px solid black;">0 Dropout WER (%)&darr;</th>
      <th colspan="4" style="padding: 5px; border-bottom: 1px solid black;">0.5 Dropout WER (%)&darr;</th>
    </tr>
    <tr style="background-color: #f2f2f2;">
      <th style="padding: 5px; border-bottom: 1px solid black;">TED-LIUM 3</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-clean</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-other</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">LS-Ave.</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">TED-LIUM 3</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-clean</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-other</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">LS-Ave.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 5px; border-bottom: 1px solid black;">0</td>
      <td style="padding: 5px; text-align: left; border-bottom: 1px solid black;">- / -</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">7.89</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">4.79</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">9.83</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">7.310</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">-</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">-</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">-</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">-</td>
    </tr>
    <tr>
      <td rowspan="5" style="vertical-align: middle; border-bottom: 1px solid black; padding: 5px;">1</td>
      <td style="padding: 5px; text-align: left;">GT / GT</td>
      <td style="padding: 5px;">5.6</td>
      <td style="padding: 5px;">4.49</td>
      <td style="padding: 5px;">10.36</td>
      <td style="padding: 5px;">7.425</td>
      <td style="padding: 5px;">7.89</td>
      <td style="padding: 5px;">4.31</td>
      <td style="padding: 5px;">9.68</td>
      <td style="padding: 5px;">6.995</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / GT</td>
      <td style="padding: 5px;">5.85</td>
      <td style="padding: 5px;"><b>4.54</b></td>
      <td style="padding: 5px;">10.63</td>
      <td style="padding: 5px;">7.585</td>
      <td style="padding: 5px;">7.47</td>
      <td style="padding: 5px;">4.74</td>
      <td style="padding: 5px;">9.94</td>
      <td style="padding: 5px;">7.340</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / Whisper</td>
      <td style="padding: 5px;"><b>5.62</b></td>
      <td style="padding: 5px;">4.67</td>
      <td style="padding: 5px;"><b>9.46</b></td>
      <td style="padding: 5px;"><b>7.065</b></td>
      <td style="padding: 5px;">7.21</td>
      <td style="padding: 5px;">5.37</td>
      <td style="padding: 5px;">9.96</td>
      <td style="padding: 5px;">7.665</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px;">5.69</td>
      <td style="padding: 5px;">4.71</td>
      <td style="padding: 5px;">9.57</td>
      <td style="padding: 5px;">7.140</td>
      <td style="padding: 5px;"><b>5.32</b></td>
      <td style="padding: 5px;"><b>4.56</b></td>
      <td style="padding: 5px;">9.38</td>
      <td style="padding: 5px;"><b>6.970</b></td>
    </tr>
    <tr>
      <td style="border-bottom: 1px solid black; padding: 5px; text-align: left;">&nbsp;&nbsp;+ SFT2</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">5.76</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">4.67</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">9.49</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.080</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.26</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">5.14</td>
      <td style="border-bottom: 1px solid black; padding: 5px;"><b>9.30</b></td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.220</td>
    </tr>
    <tr>
      <td rowspan="5" style="vertical-align: middle; border-bottom: 1px solid black; padding: 5px;">2</td>
      <td style="padding: 5px; text-align: left;">GT / GT</td>
      <td style="padding: 5px;">6.73</td>
      <td style="padding: 5px;">4.10</td>
      <td style="padding: 5px;">8.36</td>
      <td style="padding: 5px;">6.230</td>
      <td style="padding: 5px;">5.66</td>
      <td style="padding: 5px;">4.10</td>
      <td style="padding: 5px;">8.37</td>
      <td style="padding: 5px;">6.235</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / GT</td>
      <td style="padding: 5px;">6.89</td>
      <td style="padding: 5px;">4.85</td>
      <td style="padding: 5px;">9.88</td>
      <td style="padding: 5px;">7.365</td>
      <td style="padding: 5px;">5.59</td>
      <td style="padding: 5px;">5.15</td>
      <td style="padding: 5px;"><b>9.10</b></td>
      <td style="padding: 5px;">7.130</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / Whisper</td>
      <td style="padding: 5px;">8.15</td>
      <td style="padding: 5px;">5.57</td>
      <td style="padding: 5px;">12.00</td>
      <td style="padding: 5px;">8.785</td>
      <td style="padding: 5px;">5.47</td>
      <td style="padding: 5px;">5.14</td>
      <td style="padding: 5px;">9.50</td>
      <td style="padding: 5px;">7.320</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px;"><b>5.07</b></td>
      <td style="padding: 5px;">4.87</td>
      <td style="padding: 5px;"><b>9.51</b></td>
      <td style="padding: 5px;"><b>7.190</b></td>
      <td style="padding: 5px;"><b>5.17</b></td>
      <td style="padding: 5px;"><b>4.84</b></td>
      <td style="padding: 5px;">9.19</td>
      <td style="padding: 5px;"><b>7.015</b></td>
    </tr>
    <tr>
      <td style="border-bottom: 1px solid black; padding: 5px; text-align: left;">&nbsp;&nbsp;+ SFT2</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">6.90</td>
      <td style="border-bottom: 1px solid black; padding: 5px;"><b>4.55</b></td>
      <td style="border-bottom: 1px solid black; padding: 5px;">11.17</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.860</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">6.10</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">5.43</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">9.66</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.545</td>
    </tr>
    <tr>
      <td rowspan="5" style="vertical-align: middle; border-bottom: 1px solid black; padding: 5px;">3</td>
      <td style="padding: 5px; text-align: left;">GT / GT</td>
      <td style="padding: 5px;">7.35</td>
      <td style="padding: 5px;">4.24</td>
      <td style="padding: 5px;">8.29</td>
      <td style="padding: 5px;">6.265</td>
      <td style="padding: 5px;">10.42</td>
      <td style="padding: 5px;">4.89</td>
      <td style="padding: 5px;">10.36</td>
      <td style="padding: 5px;">7.625</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / GT</td>
      <td style="padding: 5px;">7.05</td>
      <td style="padding: 5px;"><b>5.03</b></td>
      <td style="padding: 5px;">10.68</td>
      <td style="padding: 5px;">7.855</td>
      <td style="padding: 5px;">12.62</td>
      <td style="padding: 5px;">5.28</td>
      <td style="padding: 5px;">10.93</td>
      <td style="padding: 5px;">8.105</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / Whisper</td>
      <td style="padding: 5px;">10.06</td>
      <td style="padding: 5px;">5.36</td>
      <td style="padding: 5px;">10.69</td>
      <td style="padding: 5px;">8.025</td>
      <td style="padding: 5px;">7.87</td>
      <td style="padding: 5px;">5.93</td>
      <td style="padding: 5px;">10.39</td>
      <td style="padding: 5px;">8.160</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px;"><b>5.98</b></td>
      <td style="padding: 5px;">5.60</td>
      <td style="padding: 5px;"><b>9.96</b></td>
      <td style="padding: 5px;"><b>7.780</b></td>
      <td style="padding: 5px;"><b>5.18</b></td>
      <td style="padding: 5px;"><b>4.73</b></td>
      <td style="padding: 5px;"><b>9.36</b></td>
      <td style="padding: 5px;"><b>7.045</b></td>
    </tr>
    <tr>
      <td style="border-bottom: 1px solid black; padding: 5px; text-align: left;">&nbsp;&nbsp;+ SFT2</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">9.30</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">5.22</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">10.49</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.855</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">8.01</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">6.11</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">10.20</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">8.155</td>
    </tr>
    <tr>
      <td rowspan="5" style="vertical-align: middle; border-bottom: 1px solid black; padding: 5px;">4</td>
      <td style="padding: 5px; text-align: left;">GT / GT</td>
      <td style="padding: 5px;">8.54</td>
      <td style="padding: 5px;">4.26</td>
      <td style="padding: 5px;">9.01</td>
      <td style="padding: 5px;">6.635</td>
      <td style="padding: 5px;">9.22</td>
      <td style="padding: 5px;"><b>4.75</b></td>
      <td style="padding: 5px;">10.23</td>
      <td style="padding: 5px;">7.490</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / GT</td>
      <td style="padding: 5px;">7.74</td>
      <td style="padding: 5px;">4.87</td>
      <td style="padding: 5px;">11.07</td>
      <td style="padding: 5px;">7.970</td>
      <td style="padding: 5px;">10.87</td>
      <td style="padding: 5px;"><b>4.75</b></td>
      <td style="padding: 5px;">10.23</td>
      <td style="padding: 5px;">7.490</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / Whisper</td>
      <td style="padding: 5px;">87.37</td>
      <td style="padding: 5px;"><b>4.66</b></td>
      <td style="padding: 5px;">10.81</td>
      <td style="padding: 5px;">7.735</td>
      <td style="padding: 5px;">7.81</td>
      <td style="padding: 5px;"><b>4.75</b></td>
      <td style="padding: 5px;">9.82</td>
      <td style="padding: 5px;">7.285</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px;"><b>4.93</b></td>
      <td style="padding: 5px;">4.79</td>
      <td style="padding: 5px;"><b>9.97</b></td>
      <td style="padding: 5px;"><b>7.380</b></td>
      <td style="padding: 5px;"><b>5.69</b></td>
      <td style="padding: 5px;">4.79</td>
      <td style="padding: 5px;"><b>9.25</b></td>
      <td style="padding: 5px;"><b>7.020</b></td>
    </tr>
    <tr>
      <td style="border-bottom: 1px solid black; padding: 5px; text-align: left;">&nbsp;&nbsp;+ SFT2</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">113.95</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">4.90</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">11.34</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">8.120</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">9.16</td>
      <td style="border-bottom: 1px solid black; padding: 5px;"><b>4.75</b></td>
      <td style="border-bottom: 1px solid black; padding: 5px;">9.83</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.290</td>
    </tr>
     <tr>
      <td rowspan="5" style="vertical-align: middle; border-bottom: 1px solid black; padding: 5px;">5</td>
      <td style="padding: 5px; text-align: left;">GT / GT</td>
      <td style="padding: 5px;">8.72</td>
      <td style="padding: 5px;">5.46</td>
      <td style="padding: 5px;">9.49</td>
      <td style="padding: 5px;">7.475</td>
      <td style="padding: 5px;">9.57</td>
      <td style="padding: 5px;">4.90</td>
      <td style="padding: 5px;">10.04</td>
      <td style="padding: 5px;">7.470</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / GT</td>
      <td style="padding: 5px;">10.34</td>
      <td style="padding: 5px;">5.08</td>
      <td style="padding: 5px;">10.76</td>
      <td style="padding: 5px;">7.920</td>
      <td style="padding: 5px;">8.19</td>
      <td style="padding: 5px;">5.36</td>
      <td style="padding: 5px;">11.29</td>
      <td style="padding: 5px;">8.325</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">hyp / Whisper</td>
      <td style="padding: 5px;">135.57</td>
      <td style="padding: 5px;"><b>4.59</b></td>
      <td style="padding: 5px;">10.87</td>
      <td style="padding: 5px;">7.730</td>
      <td style="padding: 5px;">8.5</td>
      <td style="padding: 5px;">4.95</td>
      <td style="padding: 5px;">9.33</td>
      <td style="padding: 5px;">7.140</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px;"><b>5.34</b></td>
      <td style="padding: 5px;">4.67</td>
      <td style="padding: 5px;"><b>9.85</b></td>
      <td style="padding: 5px;"><b>7.260</b></td>
      <td style="padding: 5px;"><b>4.96</b></td>
      <td style="padding: 5px;"><b>4.55</b></td>
      <td style="padding: 5px;"><b>9.24</b></td>
      <td style="padding: 5px;"><b>6.895</b></td>
    </tr>
    <tr>
      <td style="border-bottom: 1px solid black; padding: 5px; text-align: left;">&nbsp;&nbsp;+ SFT2</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">72.55</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">4.90</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">10.57</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.735</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">8.51</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">5.23</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">9.33</td>
      <td style="border-bottom: 1px solid black; padding: 5px;">7.280</td>
    </tr>
  </tbody>
</table>

### DPO LoRA Scaling Factor Result
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center; font-family: 'Times New Roman', Times, serif; font-size: 14px; color: #000; border: 1px solid black;">
  <caption style="caption-side: bottom; text-align: left; padding-top: 10px; font-size: 13px; color: #000;">
    Table 2: Impact of DPO LoRA scaling factor (&gamma;) during inference. TED-LIUM 3 Gap denotes the WER degradation caused by irrelevant context attacks. "Attacks/o" refers to relevant context inference, while "Attacks/w" refers to irrelevent context randomly selected from the test set.
  </caption>
  <thead>
    <tr style="border-bottom: 1px solid black;">
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">&gamma;</th>
      <th colspan="3" style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">TED-LIUM 3 (WER %) &darr;</th>
      <th colspan="3" style="padding: 5px; border-bottom: 1px solid black;">LibriSpeech (WER %) &darr;</th>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <th style="padding: 5px; border-bottom: 1px solid black;">Attacks/o</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Attacks/w</th>
      <th style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">Gap &darr;</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-clean</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-other</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Ave.</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; border-right: 1px solid black;">0</td>
      <td style="padding: 5px;">5.47</td>
      <td style="padding: 5px;">7.93</td>
      <td style="padding: 5px; border-right: 1px solid black;">2.46</td>
      <td style="padding: 5px;">5.14</td>
      <td style="padding: 5px;">9.50</td>
      <td style="padding: 5px;">7.320</td>
    </tr>
    <tr>
      <td style="padding: 5px; border-right: 1px solid black;">0.0625</td>
      <td style="padding: 5px;">5.37</td>
      <td style="padding: 5px;">7.13</td>
      <td style="padding: 5px; border-right: 1px solid black;">1.76</td>
      <td style="padding: 5px;">5.12</td>
      <td style="padding: 5px;">9.31</td>
      <td style="padding: 5px;">7.215</td>
    </tr>
    <tr>
      <td style="padding: 5px; border-right: 1px solid black;">0.125</td>
      <td style="padding: 5px;">5.11</td>
      <td style="padding: 5px;">5.76</td>
      <td style="padding: 5px; border-right: 1px solid black;">0.65</td>
      <td style="padding: 5px;">5.02</td>
      <td style="padding: 5px;">9.53</td>
      <td style="padding: 5px;">7.275</td>
    </tr>
    <tr>
      <td style="padding: 5px; border-right: 1px solid black;">0.1875</td>
      <td style="padding: 5px;"><b>5.06</b></td>
      <td style="padding: 5px;">5.69</td>
      <td style="padding: 5px; border-right: 1px solid black;">0.63</td>
      <td style="padding: 5px;"><b>4.70</b></td>
      <td style="padding: 5px;"><b>9.08</b></td>
      <td style="padding: 5px;"><b>6.890</b></td>
    </tr>
    <tr>
      <td style="padding: 5px; border-right: 1px solid black;">0.25</td>
      <td style="padding: 5px;">5.17</td>
      <td style="padding: 5px;"><b>5.63</b></td>
      <td style="padding: 5px; border-right: 1px solid black;">0.46</td>
      <td style="padding: 5px;">4.84</td>
      <td style="padding: 5px;">9.19</td>
      <td style="padding: 5px;">7.015</td>
    </tr>
    <tr>
      <td style="padding: 5px; border-right: 1px solid black;">0.375</td>
      <td style="padding: 5px;">5.55</td>
      <td style="padding: 5px;">5.73</td>
      <td style="padding: 5px; border-right: 1px solid black;"><b>0.18</b></td>
      <td style="padding: 5px;">4.85</td>
      <td style="padding: 5px;">9.63</td>
      <td style="padding: 5px;">7.240</td>
    </tr>
    <tr>
      <td style="padding: 5px; border-right: 1px solid black;">0.5</td>
      <td style="padding: 5px;">8.39</td>
      <td style="padding: 5px;">8.67</td>
      <td style="padding: 5px; border-right: 1px solid black;">0.28</td>
      <td style="padding: 5px;">6.44</td>
      <td style="padding: 5px;">12.14</td>
      <td style="padding: 5px;">9.290</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">53.26</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">57.15</td>
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">3.89</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">27.11</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">28.96</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">28.035</td>
    </tr>
  </tbody>
</table>

### Robustness to Irrelevant Context
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center; font-family: 'Times New Roman', Times, serif; font-size: 14px; color: #000; border: 1px solid black;">
  <caption style="caption-side: bottom; text-align: left; padding-top: 10px; font-size: 13px; color: #000;">
    Table 3: Robustness analysis against irrelevant context attacks on TED-LIUM 3. We evaluate model resilience by replacing the historical context with randomly sampled irrelevant text.
  </caption>
  <thead>
    <tr style="border-bottom: 1px solid black;">
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">N</th>
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">Con<sub>inf</sub> / Con<sub>train</sub></th>
      <th colspan="3" style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">0 Dropout WER (%)&darr;</th>
      <th colspan="3" style="padding: 5px; border-bottom: 1px solid black;">0.5 Dropout WER (%)&darr;</th>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <th style="padding: 5px; border-bottom: 1px solid black;">Attacks/o</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Attacks/w</th>
      <th style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">Gap &darr;</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Attacks/o</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Attacks/w</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Gap &darr;</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom: 1px solid black;">
      <td rowspan="3" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">1</td>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / GT</td>
      <td style="padding: 5px;">5.85</td>
      <td style="padding: 5px;">8.32</td>
      <td style="padding: 5px; border-right: 1px solid black;">2.47</td>
      <td style="padding: 5px;">7.47</td>
      <td style="padding: 5px;">8.82</td>
      <td style="padding: 5px;">1.35</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / Whisper</td>
      <td style="padding: 5px;"><b>5.62</b></td>
      <td style="padding: 5px;">6.27</td>
      <td style="padding: 5px; border-right: 1px solid black;">0.65</td>
      <td style="padding: 5px;">7.21</td>
      <td style="padding: 5px;">7.09</td>
      <td style="padding: 5px;"><b>-0.12</b></td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; text-align: left; border-right: 1px solid black; border-bottom: 1px solid black;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px; border-bottom: 1px solid black;">5.69</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.5</b></td>
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;"><b>-0.19</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.32</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.23</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;">-0.09</td>
    </tr>
    <tr>
      <td rowspan="3" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">2</td>
      <td style="padding: 5px; text-align: left;border-right: 1px solid black;">hyp / GT</td>
      <td style="padding: 5px;">6.89</td>
      <td style="padding: 5px;">8.43</td>
      <td style="padding: 5px; border-right: 1px solid black;">1.54</td>
      <td style="padding: 5px;">5.59</td>
      <td style="padding: 5px;">9.23</td>
      <td style="padding: 5px;">3.46</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / Whisper</td>
      <td style="padding: 5px;">8.15</td>
      <td style="padding: 5px;">10.37</td>
      <td style="padding: 5px; border-right: 1px solid black;">2.22</td>
      <td style="padding: 5px;">5.47</td>
      <td style="padding: 5px;">7.93</td>
      <td style="padding: 5px;">2.46</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; text-align: left; border-right: 1px solid black; border-bottom: 1px solid black;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.07</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>6.59</b></td>
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;"><b>1.52</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.17</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.63</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>0.46</b></td>
    </tr>
    <tr>
      <td rowspan="3" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">3</td>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / GT</td>
      <td style="padding: 5px;">7.05</td>
      <td style="padding: 5px;">7.64</td>
      <td style="padding: 5px; border-right: 1px solid black;">0.59</td>
      <td style="padding: 5px;">12.62</td>
      <td style="padding: 5px;">10.19</td>
      <td style="padding: 5px;"><b>-2.43</b></td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / Whisper</td>
      <td style="padding: 5px;">10.06</td>
      <td style="padding: 5px;">11.18</td>
      <td style="padding: 5px; border-right: 1px solid black;">1.12</td>
      <td style="padding: 5px;">7.87</td>
      <td style="padding: 5px;">9.53</td>
      <td style="padding: 5px;">1.66</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; text-align: left; border-right: 1px solid black; border-bottom: 1px solid black;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.98</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>6.24</b></td>
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;"><b>0.26</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.18</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.31</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;">0.13</td>
    </tr>
    <tr>
      <td rowspan="3" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">4</td>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / GT</td>
      <td style="padding: 5px;">7.74</td>
      <td style="padding: 5px;">9.75</td>
      <td style="padding: 5px; border-right: 1px solid black;">2.01</td>
      <td style="padding: 5px;">10.87</td>
      <td style="padding: 5px;">13.15</td>
      <td style="padding: 5px;">2.28</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / Whisper</td>
      <td style="padding: 5px;">87.37</td>
      <td style="padding: 5px;">8.8</td>
      <td style="padding: 5px; border-right: 1px solid black;"><b>-78.57</b></td>
      <td style="padding: 5px;">7.81</td>
      <td style="padding: 5px;">8.82</td>
      <td style="padding: 5px;"><b>1.01</b></td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; text-align: left; border-right: 1px solid black; border-bottom: 1px solid black;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>4.93</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.44</b></td>
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">0.51</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.69</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>7.58</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;">1.89</td>
    </tr>
    <tr>
      <td rowspan="3" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">5</td>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / GT</td>
      <td style="padding: 5px;">10.34</td>
      <td style="padding: 5px;">11.83</td>
      <td style="padding: 5px; border-right: 1px solid black;">1.49</td>
      <td style="padding: 5px;">8.19</td>
      <td style="padding: 5px;">10.41</td>
      <td style="padding: 5px;">2.22</td>
    </tr>
    <tr>
      <td style="padding: 5px; text-align: left; border-right: 1px solid black;">hyp / Whisper</td>
      <td style="padding: 5px;">135.57</td>
      <td style="padding: 5px;">10.74</td>
      <td style="padding: 5px; border-right: 1px solid black;"><b>-124.83</b></td>
      <td style="padding: 5px;">8.5</td>
      <td style="padding: 5px;">11.34</td>
      <td style="padding: 5px;">2.84</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 5px; text-align: left; border-right: 1px solid black; border-bottom: 1px solid black;">&nbsp;&nbsp;+ DPO</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.34</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>6.20</b></td>
      <td style="padding: 5px; border-right: 1px solid black; border-bottom: 1px solid black;">0.86</td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>4.96</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>5.51</b></td>
      <td style="padding: 5px; border-bottom: 1px solid black;"><b>0.55</b></td>
    </tr>
  </tbody>
</table>

### Data Selection and Inference Scaling for DPO
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center; font-family: 'Times New Roman', Times, serif; font-size: 13px; color: #000; border: 1px solid black;">
  <caption style="caption-side: bottom; text-align: left; padding-top: 10px; font-size: 13px; color: #000;">
    Table 4: Impact of Hard Negatives threshold and DPO LoRA scaling factor (&gamma;) during inference. TED-LIUM 3 Gap denotes the WER degradation caused by irrelevant context attacks. (N=2 and dropout rate=0.5)
  </caption>
  <thead>
    <tr style="border-bottom: 1px solid black;">
      <th rowspan="2" style="vertical-align: middle; padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">WER (%) Threshold</th>
      <th rowspan="2" style="vertical-align: middle; padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">&gamma;</th>
      <th colspan="3" style="padding: 4px; border-right: 1px solid black;">TED-LIUM 3 (WER %) &darr;</th>
      <th colspan="3" style="padding: 4px;">LibriSpeech (WER %) &darr;</th>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <th style="padding: 4px; border-bottom: 1px solid black;">Attacks/o</th>
      <th style="padding: 4px; border-bottom: 1px solid black;">Attacks/w</th>
      <th style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">Gap &darr;</th>
      <th style="padding: 4px; border-bottom: 1px solid black;">Test-clean</th>
      <th style="padding: 4px; border-bottom: 1px solid black;">Test-other</th>
      <th style="padding: 4px; border-bottom: 1px solid black;">Ave.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="8" style="vertical-align: middle; border-right: 1px solid black; border-bottom: 1px solid black;">5</td>
      <td style="padding: 4px; border-right: 1px solid black">0</td>
      <td>5.47</td><td>7.93</td><td style="border-right: 1px solid black;">2.46</td><td>5.14</td><td>9.50</td><td>7.320</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.0625</td>
      <td>5.40</td><td>7.02</td><td style="border-right: 1px solid black;">1.62</td><td>5.11</td><td><b>9.32</b></td><td><b>7.215</b></td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.125</td>
      <td>5.43</td><td>7.34</td><td style="border-right: 1px solid black;">1.91</td><td>5.18</td><td>9.41</td><td>7.295</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.1875</td>
      <td><b>5.04</b></td><td>6.76</td><td style="border-right: 1px solid black;">1.72</td><td><b>5.09</b></td><td>9.70</td><td>7.395</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.25</td>
      <td>5.23</td><td><b>5.99</b></td><td style="border-right: 1px solid black;">0.76</td><td>5.18</td><td>9.51</td><td>7.345</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.375</td>
      <td>6.21</td><td>6.42</td><td style="border-right: 1px solid black;"><b>0.21</b></td><td>5.49</td><td>10.19</td><td>7.840</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.5</td>
      <td>9.18</td><td>9.63</td><td style="border-right: 1px solid black;">0.45</td><td>8.58</td><td>14.65</td><td>11.615</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">50.40</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">50.97</td>
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.57</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">69.98</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">70.66</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">70.320</td>
    </tr>
    <tr>
      <td rowspan="8" style="vertical-align: middle; border-right: 1px solid black; border-bottom: 1px solid black;">10</td>
      <td style="padding: 4px; border-right: 1px solid black;">0</td>
      <td>5.47</td><td>7.93</td><td style="border-right: 1px solid black;">2.46</td><td>5.14</td><td>9.50</td><td>7.320</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.0625</td>
      <td>5.40</td><td>7.10</td><td style="border-right: 1px solid black;">1.70</td><td>5.11</td><td><b>9.30</b></td><td>7.205</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.125</td>
      <td>5.32</td><td>6.91</td><td style="border-right: 1px solid black;">1.59</td><td>5.19</td><td>9.61</td><td>7.400</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.1875</td>
      <td><b>5.04</b></td><td>6.63</td><td style="border-right: 1px solid black;">1.59</td><td>5.04</td><td>9.67</td><td>7.355</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.25</td>
      <td>5.27</td><td><b>6.00</b></td><td style="border-right: 1px solid black;">0.73</td><td><b>4.62</b></td><td>9.50</td><td><b>7.060</b></td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.375</td>
      <td>6.29</td><td>6.38</td><td style="border-right: 1px solid black;"><b>0.09</b></td><td>5.06</td><td>10.22</td><td>7.640</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.5</td>
      <td>10.22</td><td>10.87</td><td style="border-right: 1px solid black;">0.65</td><td>8.65</td><td>14.96</td><td>11.805</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">85.55</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">89.97</td>
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">4.42</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">88.93</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">85.49</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">87.210</td>
    </tr>
    <tr>
      <td rowspan="8" style="vertical-align: middle; border-right: 1px solid black; border-bottom: 1px solid black;">15</td>
      <td style="padding: 4px; border-right: 1px solid black;">0</td>
      <td>5.47</td><td>7.93</td><td style="border-right: 1px solid black;">2.46</td><td>5.14</td><td>9.50</td><td>7.320</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.0625</td>
      <td>5.40</td><td>7.43</td><td style="border-right: 1px solid black;">2.03</td><td>5.12</td><td>9.30</td><td>7.210</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.125</td>
      <td>5.15</td><td>6.63</td><td style="border-right: 1px solid black;">1.48</td><td>5.21</td><td>9.58</td><td>7.395</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.1875</td>
      <td>5.12</td><td>5.91</td><td style="border-right: 1px solid black;">0.79</td><td>4.98</td><td>9.46</td><td>7.220</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.25</td>
      <td><b>5.07</b></td><td><b>5.67</b></td><td style="border-right: 1px solid black;"><b>0.60</b></td><td><b>4.77</b></td><td><b>9.27</b></td><td><b>7.020</b></td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.375</td>
      <td>5.79</td><td>6.14</td><td style="border-right: 1px solid black;">0.35</td><td>5.04</td><td>9.84</td><td>7.440</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.5</td>
      <td>8.27</td><td>8.84</td><td style="border-right: 1px solid black;">0.57</td><td>6.99</td><td>12.89</td><td>9.940</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">33.58</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">35.59</td>
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">2.01</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">22.72</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">27.96</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">25.340</td>
    </tr>
    <tr>
      <td rowspan="8" style="vertical-align: middle; border-right: 1px solid black; border-bottom: 1px solid black;">20</td>
      <td style="padding: 4px; border-right: 1px solid black;">0</td>
      <td>5.47</td><td>7.93</td><td style="border-right: 1px solid black;">2.46</td><td>5.14</td><td>9.50</td><td>7.320</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.0625</td>
      <td>5.37</td><td>7.13</td><td style="border-right: 1px solid black;">1.76</td><td>5.12</td><td>9.31</td><td>7.215</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.125</td>
      <td>5.11</td><td>5.76</td><td style="border-right: 1px solid black;">0.65</td><td>5.02</td><td>9.53</td><td>7.275</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.1875</td>
      <td><b>5.06</b></td><td>5.69</td><td style="border-right: 1px solid black;">0.63</td><td><b>4.70</b></td><td><b>9.08</b></td><td><b>6.890</b></td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.25</td>
      <td>5.17</td><td><b>5.63</b></td><td style="border-right: 1px solid black;">0.46</td><td>4.84</td><td>9.19</td><td>7.015</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.375</td>
      <td>5.55</td><td>5.73</td><td style="border-right: 1px solid black;"><b>0.18</b></td><td>4.85</td><td>9.63</td><td>7.240</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.5</td>
      <td>8.39</td><td>8.67</td><td style="border-right: 1px solid black;">0.28</td><td>6.44</td><td>12.14</td><td>9.290</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">53.26</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">57.15</td>
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">3.89</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">27.11</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">28.96</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">28.035</td>
    </tr>
    <tr>
      <td rowspan="8" style="vertical-align: middle; border-right: 1px solid black; border-bottom: 1px solid black;">25</td>
      <td style="padding: 4px; border-right: 1px solid black;">0</td>
      <td>5.47</td><td>7.93</td><td style="border-right: 1px solid black;">2.46</td><td>5.14</td><td>9.50</td><td>7.320</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.0625</td>
      <td>5.39</td><td>7.40</td><td style="border-right: 1px solid black;">2.01</td><td>5.13</td><td>9.33</td><td>7.230</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.125</td>
      <td>5.28</td><td>6.00</td><td style="border-right: 1px solid black;">0.72</td><td>5.21</td><td>9.57</td><td>7.390</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.1875</td>
      <td><b>5.08</b></td><td>5.68</td><td style="border-right: 1px solid black;">0.60</td><td><b>4.70</b></td><td><b>9.18</b></td><td><b>6.940</b></td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.25</td>
      <td>5.28</td><td><b>5.34</b></td><td style="border-right: 1px solid black;"><b>0.06</b></td><td>4.84</td><td>9.33</td><td>7.085</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.375</td>
      <td>5.74</td><td>6.04</td><td style="border-right: 1px solid black;">0.30</td><td>5.12</td><td>10.01</td><td>7.565</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.5</td>
      <td>10.26</td>
      <td>10.50</td>
      <td style="padding: 4px; border-right: 1px solid black;">0.24</td>
      <td>8.25</td>
      <td>14.64</td>
      <td>11.445</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">31.01</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">30.48</td>
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">-0.53</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">28.03</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">35.84</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">31.935</td>
    </tr>
    <tr>
      <td rowspan="8" style="vertical-align: middle; border-right: 1px solid black; border-bottom: 1px solid black;">30</td>
      <td style="padding: 4px; border-right: 1px solid black;">0</td>
      <td>5.47</td><td>7.93</td><td style="border-right: 1px solid black;">2.46</td><td>5.14</td><td>9.50</td><td>7.320</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.0625</td>
      <td>5.32</td><td>7.38</td><td style="border-right: 1px solid black;">2.06</td><td>5.14</td><td><b>9.28</b></td><td>7.210</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.125</td>
      <td>5.12</td><td>6.06</td><td style="border-right: 1px solid black;">0.94</td><td>5.14</td><td>9.48</td><td>7.310</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.1875</td>
      <td>5.00</td><td>5.51</td><td style="border-right: 1px solid black;">0.51</td><td><b>4.70</b></td><td>9.54</td><td>7.120</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.25</td>
      <td><b>4.97</b></td><td>5.39</td><td style="border-right: 1px solid black;">0.42</td><td>4.74</td><td><b>9.28</b></td><td><b>7.010</b></td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.375</td>
      <td><b>4.97</b></td><td><b>5.20</b></td><td style="border-right: 1px solid black;"><b>0.23</b></td><td>4.83</td><td>9.33</td><td>7.080</td>
    </tr>
    <tr>
      <td style="padding: 4px; border-right: 1px solid black;">0.5</td>
      <td>5.49</td><td>5.78</td><td style="border-right: 1px solid black;">0.29</td><td>5.11</td><td>10.00</td><td>7.555</td>
    </tr>
    <tr style="border-bottom: 1px solid black;">
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.625</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">8.77</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">9.11</td>
      <td style="padding: 4px; border-right: 1px solid black; border-bottom: 1px solid black;">0.34</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">8.16</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">14.73</td>
      <td style="padding: 4px; border-bottom: 1px solid black;">11.445</td>
    </tr>
  </tbody>
</table>

# Citation
If this project is useful for your research, please cite:
'''
/@article{guo2026oracle,
  title={From Oracle to Noisy Context: Mitigating Contextual Exposure Bias in Speech-LLMs},
  author={Guo, Xiaoyong and Li, Nanjie and Zeng, Zijie and Wang, Kai and Huang, Hao and Xu, Haihua and Shi, Wei},
  journal={arXiv preprint arXiv:2603.24034},
  year={2026}
}
'''
