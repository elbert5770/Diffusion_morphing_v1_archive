using JLD2
using LinearAlgebra: norm
using SparseArrays
using DataFrames, CSV
using Images
using MPI
using Statistics

struct PixelInfo
    keys::Set{Int64}
    sp1_keys::Set{Int64}
    sp2_keys::Set{Int64}
    overlap_keys::Set{Int64}
end

function create_pixel_matrices(nx::Int64, ny::Int64)
    # Initialize matrix of PixelInfo structs with zero counts
    pixel_matrix = [PixelInfo(Set{Int64}(), Set{Int64}(), Set{Int64}(), Set{Int64}()) 
                   for i in 1:nx, j in 1:ny]
    
    return pixel_matrix
end

function update_pixel_info!(pixel_matrix, key, results_array, sp1_binary, sp2_binary, T_matrix)
    nx, ny = size(sp1_binary)
    unique_sp1_count = sum(sp1_binary .& .!sp2_binary)
    unique_sp2_count = sum(sp2_binary .& .!sp1_binary)
    
    # Pre-allocate these outside the loop
    empty_set = Set{Int64}()
    empty_dict = Dict{Int64, Float64}()
    
    for i in 1:nx, j in 1:ny
        if sp1_binary[i,j] || sp2_binary[i,j]
            pixel = pixel_matrix[i,j]
            if typeof(pixel) == Nothing
                pixel_matrix[i,j] = PixelInfo(Set{Int64}(), Set{Int64}(), Set{Int64}())
                pixel = pixel_matrix[i,j]
            end
            
            push!(pixel.keys, key)
            
       
            
            if sp1_binary[i,j] && sp2_binary[i,j]
                push!(pixel.overlap_keys, key)
                results_array[i,j] = key
            elseif sp1_binary[i,j]
                push!(pixel.sp1_keys, key)
            elseif sp2_binary[i,j]
                push!(pixel.sp2_keys, key)
            end
            # Update the unique pixel counts
            pixel_matrix[i,j] = PixelInfo(
                pixel_matrix[i,j].keys,
                pixel_matrix[i,j].sp1_keys,
                pixel_matrix[i,j].sp2_keys,
                pixel_matrix[i,j].overlap_keys
            )
        end
    end
end


function read_and_process_sframe_files(csv_path1::String)
    # Read CSV files without headers
    data1 = try
        df1 = DataFrame(CSV.File(csv_path1, header=false))
        df1[2:end, :]  # Remove first row
    catch e
        println("Error reading first CSV file: ", e)
        return nothing
    end
    
    return data1
end


function diffusion_migration(path::String,slice1::Int64,slice2::Int64,xloc::Int64,yloc::Int64,height::Int64,nx::Int64,ny::Int64,nz::Int64,step_per_layer::Int64,rank::Int64)
    @show rank,slice1
    
    csv_file = string(path,"sframe_$(slice1)_$(xloc)_$(yloc)_$(height).csv")
    # Read the files
    data1 = read_and_process_sframe_files(csv_file)
    output_matrix = zeros(Int64, nx, ny, 4)
    for i in 1:nx, j in 1:ny, iter in 1:4
        output_matrix[i,j,iter] = data1[i,j]
    end

    filename = string(path, "result_dict_", slice1, "_", xloc, "_", yloc, "_", height, ".jld2")
    @load filename result_dict group_dict component_mapping combined_components num_matches


    results_array = zeros(Int64, nx, ny)

    filename = string(path,"sparse_arrays_",slice1,"_",xloc,"_",yloc,"_",height,".jld2")
    @load filename sparse_arrays sparse_arrays_components
    sparse_arrays1 = sparse_arrays
    

    filename = string(path,"sparse_arrays_",slice2,"_",xloc,"_",yloc,"_",height,".jld2")
    @load filename sparse_arrays sparse_arrays_components
    sparse_arrays2 = sparse_arrays
    

    sp1_binary = zeros(Bool, nx, ny)
    sp2_binary = zeros(Bool, nx, ny)
    T_matrix = zeros(Float64, nx, ny)
    T_matrix2 = zeros(Float64, nx, ny)
    pixel_matrix = create_pixel_matrices(nx, ny)
    substitution_counts = Dict{Tuple{Int64, Int64, Int64}, Int64}()
    # Before the binary mask filling, add this processing:
    for (key_idx, (key, (groups, mapping_array))) in enumerate(group_dict)
      
        for (group_id, matches) in groups
            fill!(sp1_binary, false)
            fill!(sp2_binary, false)
            for match in matches
                # @show key,group_id,match,groups,mapping_array
                
                # Get components from mapping array
                mapping = mapping_array[:, match]
               
                comp1 = mapping[1]  # First component
                comp2 = mapping[2]  # Second component
               
                if comp1 != 0
                    # Extract rows and cols from sparse_arrays1
                    sp1 = sparse_arrays1[key,comp1]
                    rows1, cols1, _ = findnz(sp1)
                    
                    
                    # Fill binary mask for slice 1
                    for (row, col) in zip(rows1, cols1)
                        sp1_binary[row, col] = true
                    end
                end
                
                if comp2 != 0
                    # Extract rows and col2s from sparse_arrays2
                    sp2 = sparse_arrays2[key,comp2]
                    rows2, cols2, _ = findnz(sp2)
                    
                    # Fill binary mask for slice 2
                    for (row, col) in zip(rows2, cols2)
                        sp2_binary[row, col] = true
                    end
                end

  
            end  

      
            # @show key,slice1,slice2
            sampled_T = result_dict[key][slice1*5+1]
           
            fill!(T_matrix, 0.0)
            rowsT, colsT, valuesT = findnz(sampled_T)
            # Fill binary mask for slice 2
            for idx in axes(rowsT, 1)
                T_matrix[rowsT[idx], colsT[idx]] = valuesT[idx]
            end
           
            update_pixel_info!(pixel_matrix, key, results_array, sp1_binary, sp2_binary, T_matrix)

            
            
            # display(slice_fig)
            ∇x = zeros(nx,ny)
            ∇y = zeros(nx,ny)
            
            # Calculate 2D gradients at the analysis slice
            for i in 1:nx
                for j in 1:ny
                    if sp2_binary[i,j] && !sp1_binary[i,j]
                        # Handle edge cases with one-sided derivatives
                        if i <= 2  # Left edge
                            ∇x[i,j] = (-T_matrix[i+2,j] + 4*T_matrix[i+1,j] - 3*T_matrix[i,j]) / 2
                        elseif i >= nx-1  # Right edge
                            ∇x[i,j] = (3*T_matrix[i,j] - 4*T_matrix[i-1,j] + T_matrix[i-2,j]) / 2
                        else  # Interior points
                            ∇x[i,j] = (-T_matrix[i+2,j] + 8*T_matrix[i+1,j] - 8*T_matrix[i-1,j] + T_matrix[i-2,j]) / 12
                        end

                        if j <= 2  # Bottom edge
                            ∇y[i,j] = (-T_matrix[i,j+2] + 4*T_matrix[i,j+1] - 3*T_matrix[i,j]) / 2
                        elseif j >= ny-1  # Top edge
                            ∇y[i,j] = (3*T_matrix[i,j] - 4*T_matrix[i,j-1] + T_matrix[i,j-2]) / 2
                        else  # Interior points
                            ∇y[i,j] = (-T_matrix[i,j+2] + 8*T_matrix[i,j+1] - 8*T_matrix[i,j-1] + T_matrix[i,j-2]) / 12
                        end
                        
                    end
                    
                    
                end
            end

         
            
            location_temp = zeros(Int64,2)
            exact_location = zeros(Float64,2)
            alpha = 1.0
            mask_sp2_unique = zeros(Bool,nx,ny)
            scale = 5.0
            path_info_sp2 = Dict{Tuple{Int64,Int64}, Dict{Tuple{Int64,Int64},Tuple{Float64,Int64,Int64}}}()
            for i in 1:nx
                for j in 1:ny
 
                    if sp2_binary[i,j] && !sp1_binary[i,j]
                        mask_sp2_unique[i,j] = true
                        location_temp[1] = i
                        location_temp[2] = j
                        exact_location[1] = i
                        exact_location[2] = j
                        path_length = 0.0
                        grad_x = 0.0
                        grad_y = 0.0
                        
                            # Initialize arrays to store path coordinates
                            path_x = Float64[exact_location[1]]
                            path_y = Float64[exact_location[2]]
                            counter = 0
                            
                            while location_temp[1] > 0 && location_temp[1] <= nx && 
                                location_temp[2] > 0 && location_temp[2] <= ny &&
                                counter < 2000 &&
                                isempty(pixel_matrix[location_temp[1], location_temp[2]].overlap_keys)

                                
                                
                                
                                # Get fractional parts for interpolation weights
                                fx = exact_location[1] - floor(exact_location[1])
                                fy = exact_location[2] - floor(exact_location[2])
                                
                                # Get the four surrounding grid points
                                x0 = max(1, Int64(floor(exact_location[1])))
                                x1 = min(nx, Int64(ceil(exact_location[1])))
                                y0 = max(1, Int64(floor(exact_location[2])))
                                y1 = min(ny, Int64(ceil(exact_location[2])))
                               
                                    # Bilinear interpolation
                                    alpha = 1.0
                                    grad_x = (1-fx)*(1-fy)*∇x[x0,y0] + 
                                            fx*(1-fy)*∇x[x1,y0] + 
                                            (1-fx)*fy*∇x[x0,y1] + 
                                            fx*fy*∇x[x1,y1]
                                    
                                    grad_y = (1-fx)*(1-fy)*∇y[x0,y0] + 
                                            fx*(1-fy)*∇y[x1,y0] + 
                                            (1-fx)*fy*∇y[x0,y1] + 
                                            fx*fy*∇y[x1,y1]
                                  
                                    exact_location[1] = exact_location[1] + alpha*grad_x
                                    exact_location[2] = exact_location[2] + alpha*grad_y
                                    # path_length = path_length + norm([grad_x, grad_y])
                                    path_length = path_length + norm([grad_x, grad_y])
                                    # Store the new location
                                    push!(path_x, exact_location[1])
                                    push!(path_y, exact_location[2])
                                    location_temp[1] = Int64(round(exact_location[1]))
                                    location_temp[2] = Int64(round(exact_location[2]))
                                    
                                # end
                                
                                counter += 1
                            end
               
                            if !haskey(path_info_sp2, (location_temp[1], location_temp[2]))
                                path_info_sp2[(location_temp[1], location_temp[2])] = Dict{Tuple{Int64,Int64},Tuple{Float64,Int64,Int64}}()
                            end
                            path_info_sp2[(location_temp[1], location_temp[2])][(i,j)] = (path_length, key, group_id)
                            
                    end
                end
            end
            
            
            for i in 1:nx
                for j in 1:ny
                    if haskey(path_info_sp2, (i,j))
                        max_path_length = maximum(path_length for (path_length, _, _) in values(path_info_sp2[(i,j)]))
                        
                        # Now iterate through all paths at this point and mark short ones
                        for ((start_i, start_j), (path_length, local_key, group_id)) in path_info_sp2[(i,j)]
                            
                           if max_path_length > 0.0
                           
                                for iter in 1:4
                                    if ceil(Int, 5 * path_length / max_path_length) <= iter
                                        output_matrix[start_i,start_j,iter] = local_key

                                    end
                                end
                            end
                        end

                    end
                    
                end
            end
            
                
        end
           
        
        
            
        
    end 
    for iter in 1:4
        filename = string(path,"zframe_",slice1*5+iter,"_$(xloc)_$(yloc)_$(height)",".csv")
        CSV.write(filename, DataFrame(output_matrix[:,:,iter], :auto))
    end
    
  
    src = string(path, "sframe_", slice1, "_$(xloc)_$(yloc)_$(height).csv")
    dst = string(path, "zframe_", slice1*5, "_$(xloc)_$(yloc)_$(height).csv")
    cp(src, dst, force=true)  
    return
end

function job_queue(path::String,start_slice::Int64,end_slice::Int64,xloc::Int64,yloc::Int64,height::Int64,nx::Int64,ny::Int64,nz::Int64,step_per_layer::Int64,comm::MPI.Comm,rank::Int64,world_size::Int64)

    nworkers = world_size - 1

    root = 0

    MPI.Barrier(comm)
    T = eltype(start_slice)
    N = end_slice-start_slice
    send_mesg = Array{T}(undef, 1)
    recv_mesg = Array{T}(undef, 1)

    if rank == root # I am root

        idx_recv = 0
        idx_sent = 1

        new_data = Array{T}(undef, N)*0
        # Array of workers requests
        sreqs_workers = Array{MPI.Request}(undef,nworkers)
        # -1 = start, 0 = channel not available, 1 = channel available
        status_workers = ones(nworkers).*-1

        # Send message to workers
        for dst in 1:nworkers
            if idx_sent > N
                break
            end
            send_mesg[1] = idx_sent
            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
            idx_sent += 1
            # if mod(idx_sent,10) == 0
            #     println("Progress $idx_sent / $N")
            # end
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            # print("Root: Sent number $(send_mesg[1]) to Worker $dst\n")
        end

        # Send and receive messages until all elements are added
        while idx_recv != N
            # Check to see if there is an available message to receive
            for dst in 1:nworkers
                if status_workers[dst] == 0
                    flag = MPI.Test(sreqs_workers[dst])
                    if flag
                        status_workers[dst] = 1
                    end
                end
            end
            for dst in 1:nworkers
                if status_workers[dst] == 1
                    ismessage = MPI.Iprobe(comm; source=dst, tag=dst+42)
                    if ismessage
                        # Receives message
                        MPI.Recv!(recv_mesg, comm; source=dst, tag=dst+42)
                        idx_recv += 1
                        new_data[idx_recv] = recv_mesg[1]
                        # print("Root: Received number $(recv_mesg[1]) from Worker $dst\n")
                        if idx_sent <= N
                            send_mesg[1] = idx_sent
                            # Sends new message
                            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
                            idx_sent += 1
                            # if mod(idx_sent,10) == 0
                            #     println("Progress $idx_sent / $N")
                            # end
                            sreqs_workers[dst] = sreq
                            status_workers[dst] = 0
                            # print("Root: Sent number $(send_mesg[1]) to Worker $dst\n")
                        end
                    end
                end
            end
        end

        for dst in 1:nworkers
            # Termination message to worker
            send_mesg[1] = -1
            sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=dst+32)
            sreqs_workers[dst] = sreq
            status_workers[dst] = 0
            # print("Root: Finish Worker $dst\n")
        end

        MPI.Waitall(sreqs_workers)
        # print("Root: New data = $new_data\n")
    else # If rank == worker
        # -1 = start, 0 = channel not available, 1 = channel available
        status_worker = -1
        sreqs_workers = Array{MPI.Request}(undef,1)
        while true
            if status_worker != 0
                ismessage = MPI.Iprobe(comm; source=root, tag=rank+32)

                if ismessage
                    # Receives message
                    MPI.Recv!(recv_mesg, comm; source=root, tag=rank+32)
                    # Termination message from root
                    if recv_mesg[1] == -1
                        # print("Worker $rank: Finish\n")
                        break
                    end
                    # print("Worker $rank: Received number $(recv_mesg[1]) from root\n")
                    slice1 = start_slice + recv_mesg[1] - 1
                    slice2 = slice1 + 1
                    
                    
                    diffusion_migration(path,slice1,slice2,xloc,yloc,height,nx,ny,nz,step_per_layer,rank)
                    send_mesg[1] = recv_mesg[1]
                    sreq = MPI.Isend(send_mesg[1], comm; dest=root, tag=rank+42)
                    sreqs_workers[1] = sreq
                    status_worker = 0
                end
            else
                # Check to see if there is an available message to receive
                flag = MPI.Test(sreqs_workers[1])
                if flag
                    status_worker = 1
                end
               
            end
        end
    end
    return

end

function global_setup(path,start_slice,end_slice,xloc,yloc,height,nx,ny,nz,step_per_layer)
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    if world_size <= 1
        println("World size must be greater than 1")
        return
    end
    
    job_queue(path,start_slice,end_slice,xloc,yloc,height,nx,ny,nz,step_per_layer,comm,rank,world_size)
    if rank == 0
        src = string(path, "sframe_", end_slice, "_$(xloc)_$(yloc)_$(height).csv")
        dst = string(path, "zframe_", end_slice*5, "_$(xloc)_$(yloc)_$(height).csv")
        cp(src, dst, force=true)  
    end

end

sim_number = parse(Int64,ARGS[1])
path = string(@__DIR__, "/../Neuron_transport_data/")
filename = string(@__DIR__,"/Simulation_settings_4.0.csv")
settings = CSV.File(filename) |> Tables.matrix

        
xloc::Int64 = settings[sim_number,2]
yloc::Int64 = settings[sim_number,3]
start_slice::Int64 = settings[sim_number,4]
height::Int64 = settings[sim_number,5]
slices = Int64(height*25)
end_slice = start_slice + slices


step_per_layer = 5
nx, ny, nz = 501, 501, step_per_layer*5
# Run the program

global_setup(path,start_slice,end_slice,xloc,yloc,height,nx,ny,nz,step_per_layer)

