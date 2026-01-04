# Contextual_Speech_LLMs
Contextual Speech LLMs

## Main Result
<table border="1" style="border-collapse: collapse; width: 100%; text-align: center; font-family: Arial, sans-serif; font-size: 14px;">
  <caption style="caption-side: bottom; text-align: left; padding-top: 10px; font-size: 13px; color: #000;">
    Table 1: WER comparison on TED-LIUM 3 and out-of-domain Librispeech dataset across different context window sizes (<i>N</i>). The column <b>Con<sub>inf</sub> / Con<sub>train</sub></b> specifies the source of history used during inference and training, respectively. <b>hyp</b> denotes using the model's own predictions as history during inference. Regarding training configuration, <b>GT</b> uses ground-truth history, while <b>Whisper</b> indicates the model was trained using context decoded by Whisper to simulate historical errors. <b>+ DPO</b> and <b>+ SFT2</b> are additional fine-tuning stages applied to the SFT model.
  </caption>
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">N</th>
      <th rowspan="2" style="vertical-align: middle; padding: 5px; border-bottom: 1px solid black;">Con<sub>inf</sub>/Con<sub>train</sub></th>
      <th colspan="4" style="padding: 5px; border-bottom: 1px solid black;">0 Dropout WER (%)&darr;</th>
      <th colspan="4" style="padding: 5px; border-bottom: 1px solid black;">0.5 Dropout WER (%)&darr;</th>
    </tr>
    <tr style="background-color: #f2f2f2;">
      <th style="padding: 5px; border-bottom: 1px solid black;">TED</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-clean</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">Test-other</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">LS-Ave.</th>
      <th style="padding: 5px; border-bottom: 1px solid black;">TED</th>
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
