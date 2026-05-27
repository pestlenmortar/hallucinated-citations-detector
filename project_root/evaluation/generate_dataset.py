import csv
import random
import os

random.seed(42)

OUT_PATH = os.path.join(os.path.dirname(__file__), "datasets", "test_citations.csv")

VALID_PAPERS = [
    {"title": "Attention is All You Need", "authors": "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin", "venue": "Advances in Neural Information Processing Systems", "vol": "30", "no": "", "pp": "5998--6008", "year": 2017, "doi": "10.48550/arXiv.1706.03762"},
    {"title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "authors": "J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova", "venue": "Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics", "vol": "", "no": "", "pp": "4171--4186", "year": 2019, "doi": "10.18653/v1/N19-1423"},
    {"title": "Deep Residual Learning for Image Recognition", "authors": "K. He, X. Zhang, S. Ren, and J. Sun", "venue": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition", "vol": "", "no": "", "pp": "770--778", "year": 2016, "doi": "10.1109/CVPR.2016.90"},
    {"title": "Generative Adversarial Nets", "authors": "I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio", "venue": "Advances in Neural Information Processing Systems", "vol": "27", "no": "", "pp": "2672--2680", "year": 2014, "doi": ""},
    {"title": "ImageNet Classification with Deep Convolutional Neural Networks", "authors": "A. Krizhevsky, I. Sutskever, and G. E. Hinton", "venue": "Advances in Neural Information Processing Systems", "vol": "25", "no": "", "pp": "1097--1105", "year": 2012, "doi": ""},
    {"title": "Adam: A Method for Stochastic Optimization", "authors": "D. P. Kingma and J. Ba", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2015, "doi": ""},
    {"title": "Long Short-Term Memory", "authors": "S. Hochreiter and J. Schmidhuber", "venue": "Neural Computation", "vol": "9", "no": "8", "pp": "1735--1780", "year": 1997, "doi": "10.1162/neco.1997.9.8.1735"},
    {"title": "Efficient Estimation of Word Representations in Vector Space", "authors": "T. Mikolov, K. Chen, G. Corrado, and J. Dean", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2013, "doi": ""},
    {"title": "Very Deep Convolutional Networks for Large-Scale Image Recognition", "authors": "K. Simonyan and A. Zisserman", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2015, "doi": ""},
    {"title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "authors": "S. Ioffe and C. Szegedy", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "448--456", "year": 2015, "doi": ""},
    {"title": "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "authors": "N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov", "venue": "Journal of Machine Learning Research", "vol": "15", "no": "56", "pp": "1929--1958", "year": 2014, "doi": ""},
    {"title": "Gradient-Based Learning Applied to Document Recognition", "authors": "Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner", "venue": "Proceedings of the IEEE", "vol": "86", "no": "11", "pp": "2278--2324", "year": 1998, "doi": "10.1109/5.726791"},
    {"title": "Sequence to Sequence Learning with Neural Networks", "authors": "I. Sutskever, O. Vinyals, and Q. V. Le", "venue": "Advances in Neural Information Processing Systems", "vol": "27", "no": "", "pp": "3104--3112", "year": 2014, "doi": ""},
    {"title": "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", "authors": "K. Cho, B. van Merrienboer, C. Gulcehre, D. Bahdanau, F. Bougares, H. Schwenk, and Y. Bengio", "venue": "Proceedings of the Conference on Empirical Methods in Natural Language Processing", "vol": "", "no": "", "pp": "1724--1734", "year": 2014, "doi": ""},
    {"title": "Neural Machine Translation by Jointly Learning to Align and Translate", "authors": "D. Bahdanau, K. Cho, and Y. Bengio", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2015, "doi": ""},
    {"title": "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation", "authors": "Y. Wu, M. Schuster, Z. Chen, Q. V. Le, M. Norouzi, W. Macherey, M. Krikun, Y. Cao, Q. Gao, K. Macherey, J. Klingner, A. Shah, M. Johnson, X. Liu, \u0141. Kaiser, S. Gouws, Y. Kato, T. Kudo, H. Kazawa, K. Stevens, G. Kurian, N. Patil, W. Wang, C. Young, J. Smith, J. Riesa, A. Rudnick, O. Vinyals, G. Corrado, M. Hughes, and J. Dean", "venue": "arXiv preprint arXiv:1609.08144", "vol": "", "no": "", "pp": "", "year": 2016, "doi": ""},
    {"title": "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", "authors": "K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhutdinov, R. Zemel, and Y. Bengio", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "2048--2057", "year": 2015, "doi": ""},
    {"title": "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "authors": "S. Ren, K. He, R. Girshick, and J. Sun", "venue": "Advances in Neural Information Processing Systems", "vol": "28", "no": "", "pp": "91--99", "year": 2015, "doi": ""},
    {"title": "You Only Look Once: Unified, Real-Time Object Detection", "authors": "J. Redmon, S. Divvala, R. Girshick, and A. Farhadi", "venue": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition", "vol": "", "no": "", "pp": "779--788", "year": 2016, "doi": "10.1109/CVPR.2016.91"},
    {"title": "Mask R-CNN", "authors": "K. He, G. Gkioxari, P. Dollar, and R. Girshick", "venue": "Proceedings of the IEEE International Conference on Computer Vision", "vol": "", "no": "", "pp": "2961--2969", "year": 2017, "doi": "10.1109/ICCV.2017.322"},
    {"title": "Recurrent Neural Network Regularization", "authors": "W. Zaremba, I. Sutskever, and O. Vinyals", "venue": "arXiv preprint arXiv:1409.2329", "vol": "", "no": "", "pp": "", "year": 2014, "doi": ""},
    {"title": "Distributed Representations of Words and Phrases and their Compositionality", "authors": "T. Mikolov, I. Sutskever, K. Chen, G. Corrado, and J. Dean", "venue": "Advances in Neural Information Processing Systems", "vol": "26", "no": "", "pp": "3111--3119", "year": 2013, "doi": ""},
    {"title": "GloVe: Global Vectors for Word Representation", "authors": "J. Pennington, R. Socher, and C. D. Manning", "venue": "Proceedings of the Conference on Empirical Methods in Natural Language Processing", "vol": "", "no": "", "pp": "1532--1543", "year": 2014, "doi": ""},
    {"title": "Convolutional Neural Networks for Sentence Classification", "authors": "Y. Kim", "venue": "Proceedings of the Conference on Empirical Methods in Natural Language Processing", "vol": "", "no": "", "pp": "1746--1751", "year": 2014, "doi": ""},
    {"title": "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank", "authors": "R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts", "venue": "Proceedings of the Conference on Empirical Methods in Natural Language Processing", "vol": "", "no": "", "pp": "1631--1642", "year": 2013, "doi": ""},
    {"title": "A Neural Algorithm of Artistic Style", "authors": "L. A. Gatys, A. S. Ecker, and M. Bethge", "venue": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition", "vol": "", "no": "", "pp": "2414--2423", "year": 2016, "doi": ""},
    {"title": "Generating Text with Recurrent Neural Networks", "authors": "I. Sutskever, J. Martens, and G. E. Hinton", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1017--1024", "year": 2011, "doi": ""},
    {"title": "Playing Atari with Deep Reinforcement Learning", "authors": "V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra, and M. Riedmiller", "venue": "arXiv preprint arXiv:1312.5602", "vol": "", "no": "", "pp": "", "year": 2013, "doi": ""},
    {"title": "Human-Level Control through Deep Reinforcement Learning", "authors": "V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis", "venue": "Nature", "vol": "518", "no": "7540", "pp": "529--533", "year": 2015, "doi": "10.1038/nature14236"},
    {"title": "Mastering the Game of Go with Deep Neural Networks and Tree Search", "authors": "D. Silver, A. Huang, C. J. Maddison, A. Guez, L. Sifre, G. van den Driessche, J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, S. Dieleman, D. Grewe, J. Nham, N. Kalchbrenner, I. Sutskever, T. Lillicrap, M. Leach, K. Kavukcuoglu, T. Graepel, and D. Hassabis", "venue": "Nature", "vol": "529", "no": "7587", "pp": "484--489", "year": 2016, "doi": "10.1038/nature16961"},
    {"title": "Deep Learning", "authors": "Y. LeCun, Y. Bengio, and G. Hinton", "venue": "Nature", "vol": "521", "no": "7553", "pp": "436--444", "year": 2015, "doi": "10.1038/nature14539"},
    {"title": "Random Forests", "authors": "L. Breiman", "venue": "Machine Learning", "vol": "45", "no": "1", "pp": "5--32", "year": 2001, "doi": "10.1023/A:1010933404324"},
    {"title": "Support-Vector Networks", "authors": "C. Cortes and V. Vapnik", "venue": "Machine Learning", "vol": "20", "no": "3", "pp": "273--297", "year": 1995, "doi": "10.1007/BF00994018"},
    {"title": "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise", "authors": "M. Ester, H.-P. Kriegel, J. Sander, and X. Xu", "venue": "Proceedings of the International Conference on Knowledge Discovery and Data Mining", "vol": "", "no": "", "pp": "226--231", "year": 1996, "doi": ""},
    {"title": "The PageRank Citation Ranking: Bringing Order to the Web", "authors": "L. Page, S. Brin, R. Motwani, and T. Winograd", "venue": "Stanford Technical Report", "vol": "", "no": "", "pp": "", "year": 1999, "doi": ""},
    {"title": "MapReduce: Simplified Data Processing on Large Clusters", "authors": "J. Dean and S. Ghemawat", "venue": "Communications of the ACM", "vol": "51", "no": "1", "pp": "107--113", "year": 2008, "doi": "10.1145/1327452.1327492"},
    {"title": "Bigtable: A Distributed Storage System for Structured Data", "authors": "F. Chang, J. Dean, S. Ghemawat, W. C. Hsieh, D. A. Wallach, M. Burrows, T. Chandra, A. Fikes, and R. E. Gruber", "venue": "ACM Transactions on Computer Systems", "vol": "26", "no": "2", "pp": "1--26", "year": 2008, "doi": "10.1145/1365815.1365816"},
    {"title": "TensorFlow: A System for Large-Scale Machine Learning", "authors": "M. Abadi, P. Barham, J. Chen, Z. Chen, A. Davis, J. Dean, M. Devin, S. Ghemawat, G. Irving, M. Isard, M. Kudlur, J. Levenberg, R. Monga, S. Moore, D. G. Murray, B. Steiner, P. Tucker, V. Vasudevan, P. Warden, M. Wicke, Y. Yu, and X. Zheng", "venue": "Proceedings of the USENIX Symposium on Operating Systems Design and Implementation", "vol": "", "no": "", "pp": "265--283", "year": 2016, "doi": ""},
    {"title": "Scikit-learn: Machine Learning in Python", "authors": "F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay", "venue": "Journal of Machine Learning Research", "vol": "12", "no": "", "pp": "2825--2830", "year": 2011, "doi": ""},
    {"title": "ImageNet: A Large-Scale Hierarchical Image Database", "authors": "J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei", "venue": "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition", "vol": "", "no": "", "pp": "248--255", "year": 2009, "doi": "10.1109/CVPR.2009.5206848"},
    {"title": "Caffe: Convolutional Architecture for Fast Feature Embedding", "authors": "Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. Girshick, S. Guadarrama, and T. Darrell", "venue": "Proceedings of the ACM International Conference on Multimedia", "vol": "", "no": "", "pp": "675--678", "year": 2014, "doi": "10.1145/2647868.2654889"},
    {"title": "PyTorch: An Imperative Style, High-Performance Deep Learning Library", "authors": "A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala", "venue": "Advances in Neural Information Processing Systems", "vol": "32", "no": "", "pp": "8024--8035", "year": 2019, "doi": ""},
    {"title": "Graph Attention Networks", "authors": "P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Bengio", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2018, "doi": ""},
    {"title": "Semi-Supervised Classification with Graph Convolutional Networks", "authors": "T. N. Kipf and M. Welling", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2017, "doi": ""},
    {"title": "Variational Autoencoders", "authors": "D. P. Kingma and M. Welling", "venue": "arXiv preprint arXiv:1312.6114", "vol": "", "no": "", "pp": "", "year": 2013, "doi": ""},
    {"title": "Generative Adversarial Text to Image Synthesis", "authors": "S. Reed, Z. Akata, X. Yan, L. Logeswaran, B. Schiele, and H. Lee", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1060--1069", "year": 2016, "doi": ""},
    {"title": "Pixel Recurrent Neural Networks", "authors": "A. van den Oord, N. Kalchbrenner, and K. Kavukcuoglu", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1747--1756", "year": 2016, "doi": ""},
    {"title": "WaveNet: A Generative Model for Raw Audio", "authors": "A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu", "venue": "arXiv preprint arXiv:1609.03499", "vol": "", "no": "", "pp": "", "year": 2016, "doi": ""},
    {"title": "Deep Speech 2: End-to-End Speech Recognition in English and Mandarin", "authors": "D. Amodei, S. Ananthanarayanan, R. Anubhai, J. Bai, E. Battenberg, C. Case, J. Casper, B. Catanzaro, Q. Cheng, G. Chen, J. Chen, J. Chen, Z. Chen, M. Chrzanowski, A. Coates, G. Diamos, K. Ding, N. Du, E. Elsen, J. Engel, W. Fang, L. Fan, C. Fougner, L. Gao, C. Gong, A. Hannun, T. Han, L. Johannes, B. Jiang, C. Ju, B. Jun, P. LeGresley, L. Lin, J. Liu, Y. Liu, W. Li, X. Li, D. Ma, R. Narang, A. Ng, S. Ozair, Y. Peng, R. Prenger, S. Qian, Z. Quan, J. Raiman, V. Rao, S. Satheesh, D. Seetapun, S. Sengupta, K. Srinet, A. Sriram, H. Tang, L. Tang, C. Wang, J. Wang, K. Wang, Y. Wang, Z. Wang, Z. Wang, S. Wu, L. Wei, B. Xiao, W. Xie, Y. Xie, D. Yogatama, B. Yuan, J. Zhan, and Z. Zhu", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "173--182", "year": 2016, "doi": ""},
    {"title": "End-to-End Memory Networks", "authors": "S. Sukhbaatar, J. Weston, R. Fergus, and A. Szlam", "venue": "Advances in Neural Information Processing Systems", "vol": "28", "no": "", "pp": "2440--2448", "year": 2015, "doi": ""},
    {"title": "Pointer Networks", "authors": "O. Vinyals, M. Fortunato, and N. Jaitly", "venue": "Advances in Neural Information Processing Systems", "vol": "28", "no": "", "pp": "2692--2700", "year": 2015, "doi": ""},
    {"title": "Order Matters: Sequence to Sequence for Sets", "authors": "O. Vinyals, S. Bengio, and M. Kudlur", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2016, "doi": ""},
    {"title": "A Neural Conversational Model", "authors": "O. Vinyals and Q. V. Le", "venue": "arXiv preprint arXiv:1506.05869", "vol": "", "no": "", "pp": "", "year": 2015, "doi": ""},
    {"title": "Teaching Machines to Read and Comprehend", "authors": "K. M. Hermann, T. Kocisky, E. Grefenstette, L. Espeholt, W. Kay, M. Suleyman, and P. Blunsom", "venue": "Advances in Neural Information Processing Systems", "vol": "28", "no": "", "pp": "1693--1701", "year": 2015, "doi": ""},
    {"title": "Memory Networks", "authors": "J. Weston, S. Chopra, and A. Bordes", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2015, "doi": ""},
    {"title": "Learning to Generate Reviews and Discovering Sentiment", "authors": "A. Radford, R. Jozefowicz, and I. Sutskever", "venue": "arXiv preprint arXiv:1704.01444", "vol": "", "no": "", "pp": "", "year": 2017, "doi": ""},
    {"title": "Evaluating the Calibration of Modern Neural Networks", "authors": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger", "venue": "Advances in Neural Information Processing Systems", "vol": "30", "no": "", "pp": "1321--1330", "year": 2017, "doi": ""},
    {"title": "Why Should I Trust You? Explaining the Predictions of Any Classifier", "authors": "M. T. Ribeiro, S. Singh, and C. Guestrin", "venue": "Proceedings of the ACM International Conference on Knowledge Discovery and Data Mining", "vol": "", "no": "", "pp": "1135--1144", "year": 2016, "doi": "10.1145/2939672.2939778"},
    {"title": "Understanding Black-box Predictions via Influence Functions", "authors": "P. W. Koh and P. Liang", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1885--1894", "year": 2017, "doi": ""},
    {"title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks", "authors": "C. Finn, P. Abbeel, and S. Levine", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1126--1135", "year": 2017, "doi": ""},
    {"title": "Proximal Policy Optimization Algorithms", "authors": "J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov", "venue": "arXiv preprint arXiv:1707.06347", "vol": "", "no": "", "pp": "", "year": 2017, "doi": ""},
    {"title": "Continuous Control with Deep Reinforcement Learning", "authors": "T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2016, "doi": ""},
    {"title": "Dueling Network Architectures for Deep Reinforcement Learning", "authors": "Z. Wang, T. Schaul, M. Hessel, H. van Hasselt, M. Lanctot, and N. de Freitas", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1995--2003", "year": 2016, "doi": ""},
    {"title": "Asynchronous Methods for Deep Reinforcement Learning", "authors": "V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "1928--1937", "year": 2016, "doi": ""},
    {"title": "Deep Reinforcement Learning with Double Q-learning", "authors": "H. van Hasselt, A. Guez, and D. Silver", "venue": "Proceedings of the AAAI Conference on Artificial Intelligence", "vol": "", "no": "", "pp": "2094--2100", "year": 2016, "doi": ""},
    {"title": "Prioritized Experience Replay", "authors": "T. Schaul, J. Quan, I. Antonoglou, and D. Silver", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2016, "doi": ""},
    {"title": "Rainbow: Combining Improvements in Deep Reinforcement Learning", "authors": "M. Hessel, J. Modayil, H. van Hasselt, T. Schaul, G. Ostrovski, W. Dabney, D. Horgan, B. Piot, M. Azar, and D. Silver", "venue": "Proceedings of the AAAI Conference on Artificial Intelligence", "vol": "", "no": "", "pp": "3215--3222", "year": 2018, "doi": ""},
    {"title": "Attention Is All You Need in Speech Separation", "authors": "Y. Luo and N. Mesgarani", "venue": "arXiv preprint arXiv:1809.07454", "vol": "", "no": "", "pp": "", "year": 2018, "doi": ""},
    {"title": "SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition", "authors": "D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le", "venue": "Proceedings of Interspeech", "vol": "", "no": "", "pp": "2613--2617", "year": 2019, "doi": ""},
    {"title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "authors": "A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, and N. Houlsby", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2021, "doi": ""},
    {"title": "Emerging Properties in Self-Supervised Vision Transformers", "authors": "M. Caron, H. Touvron, I. Misra, H. Jegou, J. Mairal, P. Bojanowski, and A. Joulin", "venue": "Proceedings of the IEEE International Conference on Computer Vision", "vol": "", "no": "", "pp": "9630--9640", "year": 2021, "doi": "10.1109/ICCV48922.2021.00951"},
    {"title": "Denoising Diffusion Probabilistic Models", "authors": "J. Ho, A. Jain, and P. Abbeel", "venue": "Advances in Neural Information Processing Systems", "vol": "33", "no": "", "pp": "6840--6851", "year": 2020, "doi": ""},
    {"title": "Diffusion Models Beat GANs on Image Synthesis", "authors": "P. Dhariwal and A. Nichol", "venue": "Advances in Neural Information Processing Systems", "vol": "34", "no": "", "pp": "8780--8794", "year": 2021, "doi": ""},
    {"title": "Learning Transferable Visual Models From Natural Language Supervision", "authors": "A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, G. Krueger, and I. Sutskever", "venue": "International Conference on Machine Learning", "vol": "", "no": "", "pp": "8748--8763", "year": 2021, "doi": ""},
    {"title": "Language Models are Few-Shot Learners", "authors": "T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei", "venue": "Advances in Neural Information Processing Systems", "vol": "33", "no": "", "pp": "1877--1901", "year": 2020, "doi": ""},
    {"title": "GPT-4 Technical Report", "authors": "OpenAI", "venue": "arXiv preprint arXiv:2303.08774", "vol": "", "no": "", "pp": "", "year": 2023, "doi": ""},
    {"title": "LoRA: Low-Rank Adaptation of Large Language Models", "authors": "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen", "venue": "International Conference on Learning Representations", "vol": "", "no": "", "pp": "", "year": 2022, "doi": ""},
    {"title": "Training Language Models to Follow Instructions with Human Feedback", "authors": "L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe", "venue": "Advances in Neural Information Processing Systems", "vol": "35", "no": "", "pp": "27730--27744", "year": 2022, "doi": ""},
    {"title": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "authors": "J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou", "venue": "Advances in Neural Information Processing Systems", "vol": "35", "no": "", "pp": "24824--24837", "year": 2022, "doi": ""},
    {"title": "RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "authors": "P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W. tau Yih, T. Rocktaschel, S. Riedel, and D. Kiela", "venue": "Advances in Neural Information Processing Systems", "vol": "33", "no": "", "pp": "9459--9474", "year": 2020, "doi": ""},
]

REAL_AUTHORS = [
    "Y. LeCun", "Y. Bengio", "G. Hinton", "A. Krizhevsky", "I. Sutskever",
    "J. Dean", "Q. V. Le", "K. He", "R. Girshick", "J. Sun",
    "A. Vaswani", "J. Devlin", "T. Mikolov", "D. P. Kingma", "J. Ba",
    "K. Simonyan", "A. Zisserman", "S. Ren", "K. Cho", "D. Bahdanau",
    "R. Socher", "C. D. Manning", "V. Mnih", "D. Silver", "P. Abbeel",
    "S. Levine", "C. Finn", "A. Radford", "I. Goodfellow", "L. Breiman",
    "C. Cortes", "V. Vapnik", "M. Ester", "L. Page", "S. Brin",
]
REAL_VENUES = [
    "Advances in Neural Information Processing Systems",
    "International Conference on Machine Learning",
    "International Conference on Learning Representations",
    "Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition",
    "Proceedings of the IEEE International Conference on Computer Vision",
    "Journal of Machine Learning Research",
    "Nature",
    "Neural Computation",
    "Machine Learning",
    "Proceedings of the AAAI Conference on Artificial Intelligence",
    "Proceedings of the Conference on Empirical Methods in Natural Language Processing",
    "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
    "IEEE Transactions on Pattern Analysis and Machine Intelligence",
    "arXiv preprint arXiv",
]

HALLUCINATED_TITLES = [
    "Deep Attention Networks for Multimodal Sentiment Analysis in Social Media Streams",
    "Quantum-Enhanced Transformer Architectures for Large-Scale Text Classification",
    "Neural Architecture Search with Generative Adversarial Feedback Loops",
    "Capsule Graph Neural Networks for Molecular Property Prediction",
    "Self-Supervised Learning of Spatiotemporal Video Representations via Contrastive Predictive Coding with Memory Augmentation",
    "Hierarchical Multi-Scale Attention for Long Document Summarization",
    "Adversarial Meta-Learning for Cross-Domain Few-Shot Image Classification",
    "Probabilistic Neural Symbolic Reasoning for Visual Question Answering",
    "Deep Recurrent Neural Networks with Differentiable Memory Banks for Algorithmic Reasoning",
    "Attention-Based Graph Convolutional Networks for Multi-Agent Reinforcement Learning",
    "Zero-Shot Cross-Modal Retrieval with Vision-Language Transformers",
    "Efficient Fine-Tuning of Large Language Models via Adaptive Rank Selection",
    "Neural Machine Translation with Unsupervised Cross-Lingual Word Embedding Alignment",
    "Deep Reinforcement Learning for Automated Theorem Proving in First-Order Logic",
    "Temporal Difference Learning with Neural Network Function Approximators for Continuous Control in Robotics",
    "Multi-Task Learning of Part-of-Speech Tagging and Dependency Parsing with Transformer Encoders",
    "Conditional Variational Autoencoders for Text-to-Speech Synthesis with Emotion Control",
    "Differentiable Neural Computer for Visual Relational Reasoning in Video Question Answering",
    "Monte Carlo Tree Search with Deep Policy Networks for Combinatorial Optimization",
    "End-to-End Learning of Latent Representations for Multi-View Stereo Reconstruction",
    "Neural Ordinary Differential Equations with Adaptive Step Size Control for Time Series Forecasting",
    "Deep Bayesian Active Learning with Monte Carlo Dropout for Medical Image Segmentation",
    "Spectral Graph Wavelet Neural Networks for Node Classification on Large Graphs",
    "Recurrent Neural Network Transducers for Real-Time Streaming Speech Recognition with Attention Decoding",
    "Knowledge Graph Embeddings with Relation-Aware Attention Mechanisms for Link Prediction",
    "Deep One-Class Classification with Ensemble of Autoencoders for Anomaly Detection",
    "Spatial Transformer Memory Networks for Visual Navigation in Indoor Environments",
    "Federated Learning with Differential Privacy and Secure Aggregation for Healthcare Applications",
    "Neural Architecture Search with Reinforcement Learning and Network Morphisms",
    "Deep Implicit Models for Continuous-Time Dynamic Graph Representation Learning",
    "Memory-Augmented Neural Networks for Few-Shot Text Classification with Task-Adaptive Embeddings",
    "Gated Graph Sequence Neural Networks for Semantic Parsing on Knowledge Bases",
    "Multi-Modal Transformer Fusion for Visual Question Answering with Scene Graph Reasoning",
    "Deep Subspace Clustering with Self-Expressive Autoencoders for Image Segmentation",
    "Attention-Based Bidirectional LSTM Networks for Aspect-Level Sentiment Classification with Contextualized Embeddings",
    "Variational Information Bottleneck for Interpretable Deep Learning in Medical Diagnosis",
    "Generative Flow Networks for Discrete Probabilistic Modeling of Molecular Structures",
    "Deep Ensemble Learning with Stacked Generalization for Credit Risk Assessment",
    "Neural Tangent Kernel Guided Training of Deep Neural Networks for Improved Generalization",
    "Contrastive Learning of Structured World Models for Goal-Conditioned Reinforcement Learning",
]


def make_ieee(paper: dict) -> str:
    parts = [paper["authors"] + ", "]
    parts.append('"' + paper["title"] + ',"')
    parts.append(" in " + paper["venue"])
    if paper["vol"]:
        parts.append(", vol. " + paper["vol"])
    if paper["no"]:
        parts.append(", no. " + paper["no"])
    if paper["pp"]:
        parts.append(", pp. " + paper["pp"])
    parts.append(", " + str(paper["year"]))
    if paper["doi"]:
        parts.append(", doi: " + paper["doi"])
    parts.append(".")
    return "".join(parts)


def make_hallucinated_ieee(title: str) -> str:
    num_authors = random.randint(2, 5)
    authors = random.sample(REAL_AUTHORS, num_authors)
    if len(authors) == 2:
        author_str = authors[0] + " and " + authors[1]
    else:
        author_str = ", ".join(authors[:-1]) + ", and " + authors[-1]
    venue = random.choice(REAL_VENUES)
    year = random.randint(2018, 2025)
    return author_str + ', "' + title + '," in ' + venue + ", " + str(year) + "."


def corrupt_venue(paper: dict) -> str:
    p = dict(paper)
    wrong_venues = [v for v in REAL_VENUES if v != p["venue"]]
    p["venue"] = random.choice(wrong_venues)
    return make_ieee(p)


def corrupt_year(paper: dict) -> str:
    p = dict(paper)
    shift = random.choice([-3, -2, -1, 1, 2, 3])
    p["year"] = p["year"] + shift
    return make_ieee(p)


def corrupt_incomplete_authors(paper: dict) -> str:
    p = dict(paper)
    authors = p["authors"]
    if " and " in authors:
        parts = authors.split(" and ")
        last = parts[-1]
        first_part = parts[0]
        first_authors = first_part.split(", ")
        if len(first_authors) >= 3:
            p["authors"] = ", ".join(first_authors[:2]) + ", and " + last
        else:
            p["authors"] = first_authors[0] + " and " + last
    else:
        p["authors"] = authors
    return make_ieee(p)


def corrupt_missing_authors(paper: dict) -> str:
    p = dict(paper)
    authors = p["authors"]
    if " and " in authors:
        parts = authors.split(" and ")
        first_name = parts[0].split(", ")[0]
        p["authors"] = first_name + " et al."
    else:
        p["authors"] = authors.split(", ")[0] + " et al."
    return make_ieee(p)


def corrupt_title_typo(paper: dict) -> str:
    p = dict(paper)
    title = p["title"]
    replacements = {
        "All": "Al",
        "Deep": "Depp",
        "Learning": "Learrning",
        "Image": "Iamge",
        "Neural": "Neral",
        "Network": "Netwok",
        "Recurrent": "Reccurent",
        "Attention": "Atention",
        "Classification": "Clasification",
        "Machine": "Machne",
    }
    for old, new in replacements.items():
        if old in title and random.random() < 0.4:
            title = title.replace(old, new, 1)
            break
    p["title"] = title
    return make_ieee(p)


def corrupt_pages(paper: dict) -> str:
    p = dict(paper)
    if p["pp"]:
        original = p["pp"]
        parts = original.split("--")
        if len(parts) == 2:
            start = int(parts[0])
            end = int(parts[1])
            shift = random.randint(10, 100)
            p["pp"] = str(start + shift) + "--" + str(end + shift)
    return make_ieee(p)


def corrupt_missing_doi(paper: dict) -> str:
    p = dict(paper)
    p["doi"] = ""
    return make_ieee(p)


def corrupt_mixed(paper: dict) -> str:
    p = dict(paper)
    mods = []
    if random.random() < 0.5:
        wrong_venues = [v for v in REAL_VENUES if v != p["venue"]]
        p["venue"] = random.choice(wrong_venues)
        mods.append("venue")
    if random.random() < 0.5:
        p["year"] = p["year"] + random.choice([-2, -1, 1, 2])
        mods.append("year")
    if random.random() < 0.5:
        if " and " in p["authors"]:
            parts = p["authors"].split(" and ")
            first_name = parts[0].split(", ")[0]
            p["authors"] = first_name + " et al."
            mods.append("authors")
    return make_ieee(p)


def main():
    records = []
    pid = 1

    random.shuffle(VALID_PAPERS)
    valid_pool = VALID_PAPERS[:]
    valid_used = []

    for i in range(80):
        paper = valid_pool[i % len(valid_pool)]
        citation = make_ieee(paper)
        records.append({
            "citation_id": pid,
            "raw_citation": citation,
            "true_label": "VALID",
            "corruption_type": "",
            "notes": "",
        })
        pid += 1
        valid_used.append(paper)

    corruptions = {
        "wrong venue": (corrupt_venue, 10),
        "year shifted": (corrupt_year, 10),
        "incomplete authors": (corrupt_incomplete_authors, 10),
        "missing authors": (corrupt_missing_authors, 8),
        "title typo": (corrupt_title_typo, 10),
        "wrong page numbers": (corrupt_pages, 8),
        "missing DOI": (corrupt_missing_doi, 8),
    }

    partial_used = []
    for corr_type, (func, count) in corruptions.items():
        for _ in range(count):
            pool = [p for p in valid_pool if p not in partial_used]
            if not pool:
                pool = valid_pool
            paper = random.choice(pool)
            partial_used.append(paper)
            citation = func(paper)
            if corr_type == "wrong venue":
                notes = "Venue changed to a different conference/journal"
            elif corr_type == "year shifted":
                notes = "Publication year shifted by a few years"
            elif corr_type == "incomplete authors":
                notes = "Only first two authors listed"
            elif corr_type == "missing authors":
                notes = "Authors replaced with et al."
            elif corr_type == "title typo":
                notes = "Title contains a typographical error"
            elif corr_type == "wrong page numbers":
                notes = "Page numbers do not match the original"
            elif corr_type == "missing DOI":
                notes = "DOI field is empty"
            else:
                notes = ""
            records.append({
                "citation_id": pid,
                "raw_citation": citation,
                "true_label": "PARTIALLY_VALID",
                "corruption_type": corr_type,
                "notes": notes,
            })
            pid += 1

    for _ in range(16):
        pool = [p for p in valid_pool if p not in partial_used]
        if not pool:
            pool = valid_pool
        paper = random.choice(pool)
        partial_used.append(paper)
        citation = corrupt_mixed(paper)
        records.append({
            "citation_id": pid,
            "raw_citation": citation,
            "true_label": "PARTIALLY_VALID",
            "corruption_type": "mixed",
            "notes": "Multiple fields corrupted (venue, year, authors)",
        })
        pid += 1

    random.shuffle(HALLUCINATED_TITLES)
    for title in HALLUCINATED_TITLES[:40]:
        citation = make_hallucinated_ieee(title)
        records.append({
            "citation_id": pid,
            "raw_citation": citation,
            "true_label": "HALLUCINATED",
            "corruption_type": "",
            "notes": "",
        })
        pid += 1

    random.shuffle(records)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "citation_id", "raw_citation", "true_label",
            "corruption_type", "notes"
        ])
        writer.writeheader()
        writer.writerows(records)

    counts = {}
    for r in records:
        lbl = r["true_label"]
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"Generated {len(records)} records to {OUT_PATH}")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl}: {cnt}")


if __name__ == "__main__":
    main()
