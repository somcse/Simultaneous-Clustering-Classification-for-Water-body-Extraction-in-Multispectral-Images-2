from PIL import Image 
import numpy as np
b2= Image.open("C:/Users/Hp/Desktop/html/sateimages/B2_REF.tif")
b5= Image.open("C:/Users/Hp/Desktop/html/sateimages/B5_REF.tif")
b4= Image.open("C:/Users/Hp/Desktop/html/sateimages/B4_REF.tif")
b7= Image.open("C:/Users/Hp/Desktop/html/sateimages/B7_REF.tif")
b2siz = list(b2.size)
b4siz = list(b4.size)
b5siz = list(b5.size)
b7siz = list(b7.size)
b2edit = b2.load()
b4edit = b4.load()
b5edit = b5.load()
b7edit = b7.load()
dim,rows,cols=b2.size
water =  Image.new(mode="RGB", size=b2siz, color=(255, 255, 255))
wateredit = water.load()
sub_image = np.zeros((rows, cols,dim), dtype=np.float32)
for x in range(rows):
    for y in range(cols):
            awei=4 * (b2edit[x, y] - b5edit[x, y]) - (0.25 * b4edit[x, y] + 2.75 * b7edit[x, y])
            sub_image[x, y ] = [awei,x,y]
def Isodata(sub_image):
    num_clusters=6
    max_iterations=40
    min_cluster_size=15
    max_cluster_size=150
    delta=3
    iteration=1
    r1=np.random.randint(0, sub_image.shape[1], num_clusters)
    r2=np.random.randint(0, sub_image.shape[2], num_clusters)
    cluster_centers=np.zeros(num_clusters)
    rows,cols,dim=sub_image.shape
    cluster_labels=np.zeros((rows,cols,dim),dtype=np.int32)

    cluster_sizes = np.zeros(num_clusters, dtype=np.int32)
    for i in range(6):
        cluster_centers[i]=sub_image[0,r1[i],r2[i]]
 
    while iteration < max_iterations:
          
        for i in range(sub_image.shape[0]):
            for j in range(sub_image.shape[1]):
                distances = np.absolute(sub_image[i,j,0] - cluster_centers)
                cluster_labels[i,j,0] = np.argmin(distances)
                cluster_sizes[cluster_labels[i,j,0]] += 1

        new_cluster_centers = np.zeros(num_clusters, dtype=np.float32)
        print(cluster_sizes)
        for i in range(num_clusters):
            print("inside if")
            if cluster_sizes[i] > 0:
                print("inside if a")
                first_elements = sub_image[cluster_labels == i][:, 0]
                print(first_elements)
                new_cluster_centers[i] = np.mean(first_elements)
                # new_cluster_centers[i] = np.mean(sub_image[cluster_labels == i], axis=0)
            else:
                # new_cluster_centers[i] = np.zeros(num_clusters, dtype=np.float32)
                r1=np.random.randint(0, sub_image.shape[1], 1)
                r2=np.random.randint(0, sub_image.shape[2], 1)
                new_cluster_centers[i]=sub_image[0,r1,r2]
                print("inside if b")
        
    
        mse = np.sum(np.linalg.norm(new_cluster_centers - cluster_centers, axis=0)**2) / num_clusters
       
      
        for i in range(num_clusters):
            if cluster_sizes[i] < min_cluster_size or cluster_sizes[i] > max_cluster_size:
               
                if num_clusters >= 2 * i and cluster_sizes[i] > max_cluster_size:
                    
                    new_cluster_centers = np.zeros(num_clusters+1, dtype=np.float32)
                
                    new_cluster_centers[:i+1] = cluster_centers[:i+1]
                    p=i
                
                    for j in range(num_clusters-i,cluster_centers[i:].shape[0]):
                        new_cluster_centers[j]=cluster_centers[p]
                        p=p+1
                
                
                    new_cluster_centers[i+1:num_clusters-i] = (cluster_centers[i-1] + cluster_centers[i]) / 2
                
                    num_clusters += 1
                               
                
                else:
                    
                    new_cluster_centers = np.zeros(num_clusters-1, dtype=np.float32)
           
                    new_cluster_centers[:i] = cluster_centers[:i]
               
                    new_cluster_centers[i:] = cluster_centers[i+1:]
                    num_clusters -= 1
            
           
                cluster_labels = np.zeros(sub_image, dtype=np.int32)
                cluster_sizes = np.zeros(num_clusters, dtype=np.float32)
                
               
                for i in range(sub_image.shape[0]):
                    for j in range(sub_image.shape[1]):
                        distances = np.absolute(sub_image[i,j,0] - cluster_centers)
                        cluster_labels[i,j,0] = np.argmin(distances)
                        cluster_sizes[cluster_labels[i,j,0]] += 1
                    
                break
        
      
        if mse < delta:
            break
        
 
        cluster_centers = new_cluster_centers
        iteration += 1
    image_1=np.zeros((rows,cols,3),dtype=np.int32)
    image_2=np.zeros((rows,cols,3),dtype=np.int32)
    image_3=np.zeros((rows,cols,3),dtype=np.int32)
    image_4=np.zeros((rows,cols,3),dtype=np.int32)
    for x in range(rows):
        for y in range(cols):
            if cluster_labels[x,y,0]==1:
                if(awei[x,y,0]>-1.6 and awei[x,y,0]<.24):
                    image_1[x,y,0]=1
                else:
                    image_1[x,y,0]=0
            if cluster_labels[x,y,0]==2:
                if(awei[x,y,0]>-1.6 and awei[x,y,0]<.24):
                    image_2[x,y,0]=1
                else:
                    image_2[x,y,0]=0
            if cluster_labels[x,y,0]==3:
                if(awei[x,y,0]>-1.6 and awei[x,y,0]<.24):
                    image_3[x,y,0]=1
                else:
                    image_3[x,y,0]=0
            if cluster_labels[x,y,0]==4:
                if(awei[x,y,0]>-1.6 and awei[x,y,0]<.24):
                    image_4[x,y,0]=1
                else:
                    image_4[x,y,0]=0
    return image_1,image_2,image_3,image_4
def euclidean_distance(point1, point2):

  return np.linalg.norm(np.array(point1) - np.array(point2))

def extract_grids_with_distances(image):
  grid_size=5


  
  def get_grid_slice(image_shape, start, size, max_value):
      return slice(start, min(start + size, max_value))

  num_grids_h = int(np.ceil(image.shape[0] / grid_size))
  num_grids_w = int(np.ceil(image.shape[1] / grid_size))

  grids = []
  grid_centers = []
  grid_coords = []  
  for h in range(num_grids_h):
      for w in range(num_grids_w):
          x_start = get_grid_slice(image.shape, h * grid_size, grid_size, image.shape[0])
          y_start = get_grid_slice(image.shape, w * grid_size, grid_size, image.shape[1])
          grid = image[x_start, y_start]
          grids.append(grid)

          center_x = int(grid.shape[0] / 2)
          center_y = int(grid.shape[1] / 2)
          grid_centers.append((center_x, center_y))
          grid_coords.append((h * grid_size, w * grid_size))


  distances = np.zeros((len(grids), len(grids)))


  for i, center_i in enumerate(grid_centers):
        
        for j, center_j in enumerate(grid_centers):   
                    
            if i != j:

                distance = euclidean_distance(center_i, center_j)
                distances[i, j] = distance
        flat_distances = distances.flatten() 
        sorted_indices = np.argsort(flat_distances)[:3]  
        

  
        modified_grids = grids.copy()

  for i in range(len(grids)):

      flat_distances = distances[i, :]  
      sorted_indices = np.argsort(flat_distances)[:3]  
      


      all_min_grids_zero = np.all(grids[sorted_indices[1:]])[:, 0] == 0  
      all_min_grids_one = np.all(grids[sorted_indices[1:]])[:, 0] == 1 

  
      if all_min_grids_zero:
          modified_grids[i][:,0] = np.zeros_like(grids[i])  
      elif all_min_grids_one:
          modified_grids[i][:,0] = np.ones_like(grids[i])  

  return modified_grids

image_1,image_2,image_3,image_4=Isodata(sub_image)



images = [image_1, image_2, image_3, image_4]

for i, image_name in enumerate(images):
  modified_grids = extract_grids_with_distances(f"image_{i+1}")
  for grid in modified_grids:
        x,y,z=grid[0],grid[1],grid[2]
        f"image_{i+1}[x,y]=z"
  
  image_1,image_2,image_3,image_4=Isodata(f"image_{i+1}")
  for x in range(rows):
        for y in range(cols):
            if image_1[x,y,0]==1 or image_2[x,y,0]==1 or image_3[x,y,0]==1 or image_4[x,y,0]==1:
                wateredit[x,y]=(255,255,255)
            else:
                wateredit[x,y]=(0,0,0)



