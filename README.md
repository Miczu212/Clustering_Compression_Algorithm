# Cluster_Based_Compression_Algorithm
This compression algorithm, is based on dictionary like structure, which allows to reconstruct
data.
# File structure
<p> File structure is as follows

<br>• is_BMP Flag(1 byte) - defines if header is to be copied</br>
<br>• header_size (4 bytes) - size of a bmp_header</br>
<br>• original_data_size (4 byte)</br>
<br>• chunk_size (4 byte)</br>
<br>• how_many_bits_per_ID (1 byte)</br>
<br>• cluster_count (4 byte)</br>
<br>• chunk_count (4 byte)</br>
<br>• bmp_header (if BMP)</br>
<br>• centroids (n_clusters x chunk_size)</br>
<br>• saved data, as id's of closest cluster (how_many_bits_per_ID x chunk_count bits)</br>
<br>Because this algorithm is based on clusterization, data during decompression is lost, 
but that loss is dependent on ammount of clusters and their size</br></p>
