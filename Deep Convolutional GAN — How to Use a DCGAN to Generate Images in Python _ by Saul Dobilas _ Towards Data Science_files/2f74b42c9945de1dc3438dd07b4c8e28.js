document.write('<link rel="stylesheet" href="https://github.githubassets.com/assets/gist-embed-232d920b3dfe.css">')
document.write('<div id=\"gist117282716\" class=\"gist\">\n    <div class=\"gist-file\" translate=\"no\">\n      <div class=\"gist-data\">\n        <div class=\"js-gist-file-update-container js-task-list-container file-box\">\n  <div id=\"file-art053_python_001-py\" class=\"file my-2\">\n    \n    <div itemprop=\"text\" class=\"Box-body p-0 blob-wrapper data type-python  \">\n\n        \n<div class=\"js-check-bidi js-blob-code-container blob-code-content\">\n\n  <template class=\"js-file-alert-template\">\n  <div data-view-component=\"true\" class=\"flash flash-warn flash-full d-flex flex-items-center\">\n  <svg aria-hidden=\"true\" height=\"16\" viewBox=\"0 0 16 16\" version=\"1.1\" width=\"16\" data-view-component=\"true\" class=\"octicon octicon-alert\">\n    <path d=\"M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z\"><\/path>\n<\/svg>\n    <span>\n      This file contains bidirectional Unicode text that may be interpreted or compiled differently than what appears below. To review, open the file in an editor that reveals hidden Unicode characters.\n      <a class=\"Link--inTextBlock\" href=\"https://github.co/hiddenchars\" target=\"_blank\">Learn more about bidirectional Unicode characters<\/a>\n    <\/span>\n\n\n  <div data-view-component=\"true\" class=\"flash-action\">        <a href=\"{{ revealButtonHref }}\" data-view-component=\"true\" class=\"btn-sm btn\">    Show hidden characters\n<\/a>\n<\/div>\n<\/div><\/template>\n<template class=\"js-line-alert-template\">\n  <span aria-label=\"This line has hidden Unicode characters\" data-view-component=\"true\" class=\"line-alert tooltipped tooltipped-e\">\n    <svg aria-hidden=\"true\" height=\"16\" viewBox=\"0 0 16 16\" version=\"1.1\" width=\"16\" data-view-component=\"true\" class=\"octicon octicon-alert\">\n    <path d=\"M6.457 1.047c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0 1 14.082 15H1.918a1.75 1.75 0 0 1-1.543-2.575Zm1.763.707a.25.25 0 0 0-.44 0L1.698 13.132a.25.25 0 0 0 .22.368h12.164a.25.25 0 0 0 .22-.368Zm.53 3.996v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 11a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z\"><\/path>\n<\/svg>\n<\/span><\/template>\n\n  <table data-hpc class=\"highlight tab-size js-file-line-container js-code-nav-container js-tagsearch-file\" data-tab-size=\"8\" data-paste-markdown-skip data-tagsearch-lang=\"Python\" data-tagsearch-path=\"Art053_Python_001.py\">\n        <tr>\n          <td id=\"file-art053_python_001-py-L1\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"1\"><\/td>\n          <td id=\"file-art053_python_001-py-LC1\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># Tensorflow / Keras<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L2\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"2\"><\/td>\n          <td id=\"file-art053_python_001-py-LC2\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>tensorflow<\/span> <span class=pl-k>import<\/span> <span class=pl-s1>keras<\/span> <span class=pl-c># for building Neural Networks<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L3\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"3\"><\/td>\n          <td id=\"file-art053_python_001-py-LC3\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>print<\/span>(<span class=pl-s>&#39;Tensorflow/Keras: %s&#39;<\/span> <span class=pl-c1>%<\/span> <span class=pl-s1>keras<\/span>.<span class=pl-s1>__version__<\/span>) <span class=pl-c># print version<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L4\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"4\"><\/td>\n          <td id=\"file-art053_python_001-py-LC4\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>keras<\/span>.<span class=pl-s1>models<\/span> <span class=pl-k>import<\/span> <span class=pl-v>Sequential<\/span> <span class=pl-c># for assembling a Neural Network model<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L5\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"5\"><\/td>\n          <td id=\"file-art053_python_001-py-LC5\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>keras<\/span>.<span class=pl-s1>layers<\/span> <span class=pl-k>import<\/span> <span class=pl-v>Dense<\/span>, <span class=pl-v>Reshape<\/span>, <span class=pl-v>Flatten<\/span>, <span class=pl-v>Conv2D<\/span>, <span class=pl-v>Conv2DTranspose<\/span>, <span class=pl-v>ReLU<\/span>, <span class=pl-v>LeakyReLU<\/span>, <span class=pl-v>Dropout<\/span> <span class=pl-c># adding layers to the Neural Network model<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L6\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"6\"><\/td>\n          <td id=\"file-art053_python_001-py-LC6\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>tensorflow<\/span>.<span class=pl-s1>keras<\/span>.<span class=pl-s1>utils<\/span> <span class=pl-k>import<\/span> <span class=pl-s1>plot_model<\/span> <span class=pl-c># for plotting model diagram<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L7\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"7\"><\/td>\n          <td id=\"file-art053_python_001-py-LC7\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>tensorflow<\/span>.<span class=pl-s1>keras<\/span>.<span class=pl-s1>optimizers<\/span> <span class=pl-k>import<\/span> <span class=pl-v>Adam<\/span> <span class=pl-c># for model optimization <\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L8\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"8\"><\/td>\n          <td id=\"file-art053_python_001-py-LC8\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L9\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"9\"><\/td>\n          <td id=\"file-art053_python_001-py-LC9\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L10\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"10\"><\/td>\n          <td id=\"file-art053_python_001-py-LC10\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># Data manipulation<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L11\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"11\"><\/td>\n          <td id=\"file-art053_python_001-py-LC11\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>numpy<\/span> <span class=pl-k>as<\/span> <span class=pl-s1>np<\/span> <span class=pl-c># for data manipulation<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L12\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"12\"><\/td>\n          <td id=\"file-art053_python_001-py-LC12\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>print<\/span>(<span class=pl-s>&#39;numpy: %s&#39;<\/span> <span class=pl-c1>%<\/span> <span class=pl-s1>np<\/span>.<span class=pl-s1>__version__<\/span>) <span class=pl-c># print version<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L13\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"13\"><\/td>\n          <td id=\"file-art053_python_001-py-LC13\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>sklearn<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L14\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"14\"><\/td>\n          <td id=\"file-art053_python_001-py-LC14\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>print<\/span>(<span class=pl-s>&#39;sklearn: %s&#39;<\/span> <span class=pl-c1>%<\/span> <span class=pl-s1>sklearn<\/span>.<span class=pl-s1>__version__<\/span>) <span class=pl-c># print version<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L15\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"15\"><\/td>\n          <td id=\"file-art053_python_001-py-LC15\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>from<\/span> <span class=pl-s1>sklearn<\/span>.<span class=pl-s1>preprocessing<\/span> <span class=pl-k>import<\/span> <span class=pl-v>MinMaxScaler<\/span> <span class=pl-c># for scaling inputs used in the generator and discriminator<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L16\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"16\"><\/td>\n          <td id=\"file-art053_python_001-py-LC16\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L17\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"17\"><\/td>\n          <td id=\"file-art053_python_001-py-LC17\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L18\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"18\"><\/td>\n          <td id=\"file-art053_python_001-py-LC18\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># Visualization<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L19\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"19\"><\/td>\n          <td id=\"file-art053_python_001-py-LC19\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>cv2<\/span> <span class=pl-c># for ingesting images<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L20\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"20\"><\/td>\n          <td id=\"file-art053_python_001-py-LC20\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>print<\/span>(<span class=pl-s>&#39;OpenCV: %s&#39;<\/span> <span class=pl-c1>%<\/span> <span class=pl-s1>cv2<\/span>.<span class=pl-s1>__version__<\/span>) <span class=pl-c># print version<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L21\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"21\"><\/td>\n          <td id=\"file-art053_python_001-py-LC21\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>matplotlib<\/span> <\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L22\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"22\"><\/td>\n          <td id=\"file-art053_python_001-py-LC22\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>matplotlib<\/span>.<span class=pl-s1>pyplot<\/span> <span class=pl-k>as<\/span> <span class=pl-s1>plt<\/span> <span class=pl-c># or data visualizationa<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L23\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"23\"><\/td>\n          <td id=\"file-art053_python_001-py-LC23\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>print<\/span>(<span class=pl-s>&#39;matplotlib: %s&#39;<\/span> <span class=pl-c1>%<\/span> <span class=pl-s1>matplotlib<\/span>.<span class=pl-s1>__version__<\/span>) <span class=pl-c># print version<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L24\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"24\"><\/td>\n          <td id=\"file-art053_python_001-py-LC24\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>graphviz<\/span> <span class=pl-c># for showing model diagram<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L25\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"25\"><\/td>\n          <td id=\"file-art053_python_001-py-LC25\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-en>print<\/span>(<span class=pl-s>&#39;graphviz: %s&#39;<\/span> <span class=pl-c1>%<\/span> <span class=pl-s1>graphviz<\/span>.<span class=pl-s1>__version__<\/span>) <span class=pl-c># print version<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L26\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"26\"><\/td>\n          <td id=\"file-art053_python_001-py-LC26\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L27\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"27\"><\/td>\n          <td id=\"file-art053_python_001-py-LC27\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L28\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"28\"><\/td>\n          <td id=\"file-art053_python_001-py-LC28\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># Other utilities<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L29\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"29\"><\/td>\n          <td id=\"file-art053_python_001-py-LC29\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>sys<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L30\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"30\"><\/td>\n          <td id=\"file-art053_python_001-py-LC30\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-k>import<\/span> <span class=pl-s1>os<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L31\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"31\"><\/td>\n          <td id=\"file-art053_python_001-py-LC31\" class=\"blob-code blob-code-inner js-file-line\">\n<\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L32\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"32\"><\/td>\n          <td id=\"file-art053_python_001-py-LC32\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-c># Assign main directory to a variable<\/span><\/td>\n        <\/tr>\n        <tr>\n          <td id=\"file-art053_python_001-py-L33\" class=\"blob-num js-line-number js-code-nav-line-number js-blob-rnum\" data-line-number=\"33\"><\/td>\n          <td id=\"file-art053_python_001-py-LC33\" class=\"blob-code blob-code-inner js-file-line\"><span class=pl-s1>main_dir<\/span><span class=pl-c1>=<\/span><span class=pl-s1>os<\/span>.<span class=pl-s1>path<\/span>.<span class=pl-en>dirname<\/span>(<span class=pl-s1>sys<\/span>.<span class=pl-s1>path<\/span>[<span class=pl-c1>0<\/span>])<\/td>\n        <\/tr>\n  <\/table>\n<\/div>\n\n\n    <\/div>\n\n  <\/div>\n<\/div>\n\n      <\/div>\n      <div class=\"gist-meta\">\n        <a href=\"https://gist.github.com/SolClover/2f74b42c9945de1dc3438dd07b4c8e28/raw/115441fadef96831998c667420a759ed519829ac/Art053_Python_001.py\" style=\"float:right\" class=\"Link--inTextBlock\">view raw<\/a>\n        <a href=\"https://gist.github.com/SolClover/2f74b42c9945de1dc3438dd07b4c8e28#file-art053_python_001-py\" class=\"Link--inTextBlock\">\n          Art053_Python_001.py\n        <\/a>\n        hosted with &#10084; by <a class=\"Link--inTextBlock\" href=\"https://github.com\">GitHub<\/a>\n      <\/div>\n    <\/div>\n<\/div>\n')
