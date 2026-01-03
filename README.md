# Contextual_Speech_LLMs
Contextual Speech LLMs

<table>
    <thead>
        <tr>
            <th>N_Whisper</th>
            <th>Dropout</th>
            <th>WER</th>
            <th>Alpha</th>
            <th>Ordinary</th>
            <th>Attack</th>
            <th>Diff</th>
            <th>Clean</th>
            <th>Other</th>
            <th>Avg(C/O)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="14" align="center" valign="middle">1</td>
            <td align="center">0</td><td align="center">20</td><td align="center">2</td><td align="center">5.75</td><td align="center">6.01</td><td align="center">0.26</td><td align="center">4.67</td><td align="center">9.22</td><td align="center">6.945</td>
        </tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">4</td><td align="center">5.22</td><td align="center">5.84</td><td align="center">0.62</td><td align="center">4.98</td><td align="center">9.23</td><td align="center">7.105</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">6</td><td align="center">5.45</td><td align="center">5.45</td><td align="center">0.00</td><td align="center">4.74</td><td align="center">9.35</td><td align="center">7.045</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">8</td><td align="center">5.69</td><td align="center">5.50</td><td align="center">-0.19</td><td align="center">4.71</td><td align="center">9.57</td><td align="center">7.140</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">12</td><td align="center">6.51</td><td align="center">6.31</td><td align="center">-0.20</td><td align="center">5.07</td><td align="center">10.28</td><td align="center">7.675</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">16</td><td align="center">11.09</td><td align="center">11.05</td><td align="center">-0.04</td><td align="center">6.77</td><td align="center">13.19</td><td align="center">9.980</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">20</td><td align="center">26.77</td><td align="center">27.52</td><td align="center">0.75</td><td align="center">19.27</td><td align="center">28.19</td><td align="center">23.730</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">2</td><td align="center">6.44</td><td align="center">6.15</td><td align="center">-0.29</td><td align="center">4.98</td><td align="center">9.64</td><td align="center">7.310</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">4</td><td align="center">6.45</td><td align="center">6.01</td><td align="center">-0.44</td><td align="center">4.45</td><td align="center">9.28</td><td align="center">6.865</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">6</td><td align="center">4.98</td><td align="center">5.14</td><td align="center">0.16</td><td align="center">4.48</td><td align="center">9.40</td><td align="center">6.940</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">8</td><td align="center">5.32</td><td align="center">5.23</td><td align="center">-0.09</td><td align="center">4.56</td><td align="center">9.38</td><td align="center">6.970</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">12</td><td align="center">5.81</td><td align="center">6.12</td><td align="center">0.31</td><td align="center">5.17</td><td align="center">10.33</td><td align="center">7.750</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">16</td><td align="center">11.40</td><td align="center">11.42</td><td align="center">0.02</td><td align="center">7.52</td><td align="center">14.52</td><td align="center">11.020</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">20</td><td align="center">28.57</td><td align="center">28.45</td><td align="center">-0.12</td><td align="center">25.32</td><td align="center">33.32</td><td align="center">29.320</td></tr>

        <tr>
            <td rowspan="42" align="center" valign="middle">2</td>
            <td align="center">0</td><td align="center">20</td><td align="center">2</td><td align="center">6.00</td><td align="center">9.52</td><td align="center">3.52</td><td align="center">4.73</td><td align="center">11.09</td><td align="center">7.910</td>
        </tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">4</td><td align="center">5.21</td><td align="center">7.84</td><td align="center">2.63</td><td align="center">4.54</td><td align="center">10.31</td><td align="center">7.425</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">6</td><td align="center">5.23</td><td align="center">6.23</td><td align="center">1.00</td><td align="center">4.69</td><td align="center">10.00</td><td align="center">7.345</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">8</td><td align="center">5.07</td><td align="center">6.59</td><td align="center">1.52</td><td align="center">4.87</td><td align="center">9.51</td><td align="center">7.190</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">12</td><td align="center">6.08</td><td align="center">6.41</td><td align="center">0.33</td><td align="center">5.44</td><td align="center">10.60</td><td align="center">8.020</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">16</td><td align="center">12.43</td><td align="center">12.36</td><td align="center">-0.07</td><td align="center">8.37</td><td align="center">16.48</td><td align="center">12.425</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">20</td><td align="center">26.86</td><td align="center">27.53</td><td align="center">0.67</td><td align="center">24.63</td><td align="center">34.52</td><td align="center">29.575</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">2</td><td align="center">5.40</td><td align="center">7.02</td><td align="center">1.62</td><td align="center">5.11</td><td align="center">9.32</td><td align="center">7.215</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">4</td><td align="center">5.43</td><td align="center">7.34</td><td align="center">1.91</td><td align="center">5.18</td><td align="center">9.41</td><td align="center">7.295</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">6</td><td align="center">5.04</td><td align="center">6.76</td><td align="center">1.72</td><td align="center">5.09</td><td align="center">9.70</td><td align="center">7.395</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">8</td><td align="center">5.23</td><td align="center">5.99</td><td align="center">0.76</td><td align="center">5.18</td><td align="center">9.51</td><td align="center">7.345</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">12</td><td align="center">6.21</td><td align="center">6.42</td><td align="center">0.21</td><td align="center">5.49</td><td align="center">10.19</td><td align="center">7.840</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">16</td><td align="center">9.18</td><td align="center">9.63</td><td align="center">0.45</td><td align="center">8.58</td><td align="center">14.65</td><td align="center">11.615</td></tr>
        <tr><td align="center">0.5</td><td align="center">5</td><td align="center">20</td><td align="center">50.40</td><td align="center">50.97</td><td align="center">0.57</td><td align="center">69.98</td><td align="center">70.66</td><td align="center">70.320</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">2</td><td align="center">5.40</td><td align="center">7.10</td><td align="center">1.70</td><td align="center">5.11</td><td align="center">9.30</td><td align="center">7.205</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">4</td><td align="center">5.32</td><td align="center">6.91</td><td align="center">1.59</td><td align="center">5.19</td><td align="center">9.61</td><td align="center">7.400</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">6</td><td align="center">5.04</td><td align="center">6.63</td><td align="center">1.59</td><td align="center">5.04</td><td align="center">9.67</td><td align="center">7.355</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">8</td><td align="center">5.27</td><td align="center">6.00</td><td align="center">0.73</td><td align="center">4.62</td><td align="center">9.50</td><td align="center">7.060</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">12</td><td align="center">6.29</td><td align="center">6.38</td><td align="center">0.09</td><td align="center">5.06</td><td align="center">10.22</td><td align="center">7.640</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">16</td><td align="center">10.22</td><td align="center">10.87</td><td align="center">0.65</td><td align="center">8.65</td><td align="center">14.96</td><td align="center">11.805</td></tr>
        <tr><td align="center">0.5</td><td align="center">10</td><td align="center">20</td><td align="center">85.55</td><td align="center">89.97</td><td align="center">4.42</td><td align="center">88.93</td><td align="center">85.49</td><td align="center">87.210</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">2</td><td align="center">5.40</td><td align="center">7.43</td><td align="center">2.03</td><td align="center">5.12</td><td align="center">9.30</td><td align="center">7.210</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">4</td><td align="center">5.15</td><td align="center">6.63</td><td align="center">1.48</td><td align="center">5.21</td><td align="center">9.58</td><td align="center">7.395</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">6</td><td align="center">5.12</td><td align="center">5.91</td><td align="center">0.79</td><td align="center">4.98</td><td align="center">9.46</td><td align="center">7.220</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">8</td><td align="center">5.07</td><td align="center">5.67</td><td align="center">0.60</td><td align="center">4.77</td><td align="center">9.27</td><td align="center">7.020</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">12</td><td align="center">5.79</td><td align="center">6.14</td><td align="center">0.35</td><td align="center">5.04</td><td align="center">9.84</td><td align="center">7.440</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">16</td><td align="center">8.27</td><td align="center">8.84</td><td align="center">0.57</td><td align="center">6.99</td><td align="center">12.89</td><td align="center">9.940</td></tr>
        <tr><td align="center">0.5</td><td align="center">15</td><td align="center">20</td><td align="center">33.58</td><td align="center">35.59</td><td align="center">2.01</td><td align="center">22.72</td><td align="center">27.96</td><td align="center">25.340</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">2</td><td align="center">5.39</td><td align="center">7.40</td><td align="center">2.01</td><td align="center">5.13</td><td align="center">9.33</td><td align="center">7.230</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">4</td><td align="center">5.28</td><td align="center">6.00</td><td align="center">0.72</td><td align="center">5.21</td><td align="center">9.57</td><td align="center">7.390</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">6</td><td align="center">5.08</td><td align="center">5.68</td><td align="center">0.60</td><td align="center">4.70</td><td align="center">9.18</td><td align="center">6.940</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">8</td><td align="center">5.28</td><td align="center">5.34</td><td align="center">0.06</td><td align="center">4.84</td><td align="center">9.33</td><td align="center">7.085</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">12</td><td align="center">5.74</td><td align="center">6.04</td><td align="center">0.30</td><td align="center">5.12</td><td align="center">10.01</td><td align="center">7.565</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">16</td><td align="center">10.26</td><td align="center">10.50</td><td align="center">0.24</td><td align="center">8.25</td><td align="center">14.64</td><td align="center">11.445</td></tr>
        <tr><td align="center">0.5</td><td align="center">25</td><td align="center">20</td><td align="center">31.01</td><td align="center">30.48</td><td align="center">-0.53</td><td align="center">28.03</td><td align="center">35.84</td><td align="center">31.935</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">2</td><td align="center">5.32</td><td align="center">7.38</td><td align="center">2.06</td><td align="center">5.14</td><td align="center">9.28</td><td align="center">7.210</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">4</td><td align="center">5.12</td><td align="center">6.06</td><td align="center">0.94</td><td align="center">5.14</td><td align="center">9.48</td><td align="center">7.310</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">6</td><td align="center">5.00</td><td align="center">5.51</td><td align="center">0.51</td><td align="center">4.70</td><td align="center">9.54</td><td align="center">7.120</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">8</td><td align="center">4.97</td><td align="center">5.39</td><td align="center">0.42</td><td align="center">4.74</td><td align="center">9.28</td><td align="center">7.010</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">12</td><td align="center">4.97</td><td align="center">5.20</td><td align="center">0.23</td><td align="center">4.83</td><td align="center">9.33</td><td align="center">7.080</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">16</td><td align="center">5.49</td><td align="center">5.78</td><td align="center">0.29</td><td align="center">5.11</td><td align="center">10.00</td><td align="center">7.555</td></tr>
        <tr><td align="center">0.5</td><td align="center">30</td><td align="center">20</td><td align="center">8.77</td><td align="center">9.11</td><td align="center">0.34</td><td align="center">8.16</td><td align="center">14.73</td><td align="center">11.445</td></tr>

        <tr>
            <td rowspan="14" align="center" valign="middle">3</td>
            <td align="center">0</td><td align="center">20</td><td align="center">2</td><td align="center">6.65</td><td align="center">8.19</td><td align="center">1.54</td><td align="center">5.29</td><td align="center">10.17</td><td align="center">7.730</td>
        </tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">4</td><td align="center">6.40</td><td align="center">7.98</td><td align="center">1.58</td><td align="center">5.39</td><td align="center">10.46</td><td align="center">7.925</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">6</td><td align="center">5.86</td><td align="center">5.80</td><td align="center">-0.06</td><td align="center">5.17</td><td align="center">9.96</td><td align="center">7.565</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">8</td><td align="center">5.98</td><td align="center">6.24</td><td align="center">0.26</td><td align="center">5.60</td><td align="center">9.96</td><td align="center">7.780</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">12</td><td align="center">5.90</td><td align="center">5.90</td><td align="center">0.00</td><td align="center">5.30</td><td align="center">10.79</td><td align="center">8.045</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">16</td><td align="center">10.59</td><td align="center">10.80</td><td align="center">0.21</td><td align="center">8.15</td><td align="center">14.97</td><td align="center">11.560</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">20</td><td align="center">47.64</td><td align="center">50.85</td><td align="center">3.21</td><td align="center">34.07</td><td align="center">39.66</td><td align="center">36.865</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">2</td><td align="center">5.75</td><td align="center">7.17</td><td align="center">1.42</td><td align="center">5.03</td><td align="center">9.88</td><td align="center">7.455</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">4</td><td align="center">6.16</td><td align="center">6.43</td><td align="center">0.27</td><td align="center">4.98</td><td align="center">9.63</td><td align="center">7.305</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">6</td><td align="center">5.69</td><td align="center">5.76</td><td align="center">0.07</td><td align="center">4.66</td><td align="center">9.28</td><td align="center">6.970</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">8</td><td align="center">5.18</td><td align="center">5.31</td><td align="center">0.13</td><td align="center">4.73</td><td align="center">9.36</td><td align="center">7.045</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">12</td><td align="center">5.67</td><td align="center">6.00</td><td align="center">0.33</td><td align="center">4.88</td><td align="center">9.92</td><td align="center">7.400</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">16</td><td align="center">10.52</td><td align="center">10.40</td><td align="center">-0.12</td><td align="center">7.87</td><td align="center">13.55</td><td align="center">10.710</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">20</td><td align="center">41.32</td><td align="center">43.60</td><td align="center">2.28</td><td align="center">37.21</td><td align="center">41.00</td><td align="center">39.105</td></tr>

        <tr>
            <td rowspan="14" align="center" valign="middle">4</td>
            <td align="center">0</td><td align="center">20</td><td align="center">2</td><td align="center">5.98</td><td align="center">7.03</td><td align="center">1.05</td><td align="center">4.66</td><td align="center">11.12</td><td align="center">7.890</td>
        </tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">4</td><td align="center">5.25</td><td align="center">6.42</td><td align="center">1.17</td><td align="center">4.68</td><td align="center">9.99</td><td align="center">7.335</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">6</td><td align="center">5.16</td><td align="center">5.82</td><td align="center">0.66</td><td align="center">4.73</td><td align="center">9.80</td><td align="center">7.265</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">8</td><td align="center">4.93</td><td align="center">5.44</td><td align="center">0.51</td><td align="center">4.79</td><td align="center">9.97</td><td align="center">7.380</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">12</td><td align="center">5.65</td><td align="center">5.82</td><td align="center">0.17</td><td align="center">5.05</td><td align="center">10.25</td><td align="center">7.650</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">16</td><td align="center">8.96</td><td align="center">9.04</td><td align="center">0.08</td><td align="center">5.91</td><td align="center">11.88</td><td align="center">8.895</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">20</td><td align="center">28.08</td><td align="center">33.50</td><td align="center">5.42</td><td align="center">12.02</td><td align="center">20.06</td><td align="center">16.040</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">2</td><td align="center">7.13</td><td align="center">8.17</td><td align="center">1.04</td><td align="center">4.55</td><td align="center">9.42</td><td align="center">6.985</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">4</td><td align="center">7.02</td><td align="center">7.82</td><td align="center">0.80</td><td align="center">4.70</td><td align="center">9.08</td><td align="center">6.890</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">6</td><td align="center">5.56</td><td align="center">7.94</td><td align="center">2.38</td><td align="center">4.69</td><td align="center">9.13</td><td align="center">6.910</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">8</td><td align="center">5.69</td><td align="center">7.58</td><td align="center">1.89</td><td align="center">4.79</td><td align="center">9.25</td><td align="center">7.020</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">12</td><td align="center">6.22</td><td align="center">6.81</td><td align="center">0.59</td><td align="center">5.04</td><td align="center">9.72</td><td align="center">7.380</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">16</td><td align="center">9.05</td><td align="center">9.34</td><td align="center">0.29</td><td align="center">6.33</td><td align="center">11.78</td><td align="center">9.055</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">20</td><td align="center">51.13</td><td align="center">63.88</td><td align="center">12.75</td><td align="center">60.52</td><td align="center">54.45</td><td align="center">57.485</td></tr>

        <tr>
            <td rowspan="14" align="center" valign="middle">5</td>
            <td align="center">0</td><td align="center">20</td><td align="center">2</td><td align="center">98.96</td><td align="center">8.52</td><td align="center">-90.44</td><td align="center">4.58</td><td align="center">10.36</td><td align="center">7.470</td>
        </tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">4</td><td align="center">12.70</td><td align="center">6.60</td><td align="center">-6.10</td><td align="center">4.59</td><td align="center">10.19</td><td align="center">7.390</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">6</td><td align="center">6.04</td><td align="center">6.87</td><td align="center">0.83</td><td align="center">4.59</td><td align="center">9.84</td><td align="center">7.215</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">8</td><td align="center">5.34</td><td align="center">6.20</td><td align="center">0.86</td><td align="center">4.67</td><td align="center">9.85</td><td align="center">7.260</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">12</td><td align="center">6.01</td><td align="center">6.17</td><td align="center">0.16</td><td align="center">4.84</td><td align="center">9.70</td><td align="center">7.270</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">16</td><td align="center">10.59</td><td align="center">10.11</td><td align="center">-0.48</td><td align="center">5.98</td><td align="center">11.50</td><td align="center">8.740</td></tr>
        <tr><td align="center">0</td><td align="center">20</td><td align="center">20</td><td align="center">21.77</td><td align="center">21.42</td><td align="center">-0.35</td><td align="center">11.41</td><td align="center">17.27</td><td align="center">14.340</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">2</td><td align="center">7.85</td><td align="center">9.34</td><td align="center">1.49</td><td align="center">5.31</td><td align="center">9.08</td><td align="center">7.195</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">4</td><td align="center">5.99</td><td align="center">7.90</td><td align="center">1.91</td><td align="center">4.67</td><td align="center">9.15</td><td align="center">6.910</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">6</td><td align="center">5.03</td><td align="center">6.03</td><td align="center">1.00</td><td align="center">4.66</td><td align="center">9.05</td><td align="center">6.855</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">8</td><td align="center">4.96</td><td align="center">5.51</td><td align="center">0.55</td><td align="center">4.55</td><td align="center">9.24</td><td align="center">6.895</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">12</td><td align="center">5.44</td><td align="center">5.80</td><td align="center">0.36</td><td align="center">4.88</td><td align="center">9.74</td><td align="center">7.310</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">16</td><td align="center">9.06</td><td align="center">9.54</td><td align="center">0.48</td><td align="center">7.12</td><td align="center">12.79</td><td align="center">9.955</td></tr>
        <tr><td align="center">0.5</td><td align="center">20</td><td align="center">20</td><td align="center">57.88</td><td align="center">72.60</td><td align="center">14.72</td><td align="center">50.96</td><td align="center">54.77</td><td align="center">52.865</td></tr>
    </tbody>
</table>
