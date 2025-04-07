using JLD2
using LinearAlgebra
using IterativeSolvers
using SparseArrays
using DataFrames, CSV
using MPI
using IncompleteLU



function calculate_FVM_matrix(nx, ny, nz, gammax=1.0, gammay=1.0, gammaz=1.0, S=0.0)
 
    I, J, V = Int64[], Int64[], Float64[]
    
    # Helper function to convert 3D index to 1D
    function idx(i, j, k)
        return i + (j-1)*nx + (k-1)*nx*ny
    end
    
    delta_x = 1.0
    delta_y = 1.0
    delta_z = 1.0
    BC1 = 1
    BC2 = 1
    
    for k in 1:nz
        for j in 1:ny
            for i in 1:nx
                center_idx = idx(i, j, k)
                
                # Center node
                value = 0.0
                if k > 1 && k < nz
                    value += 2.0 * gammaz / delta_z
                else
                    if k == 1
                        if BC1 == 1
                            value += gammaz / (delta_z) + gammaz / (delta_z/2)
                        end
                        if BC1 == 2
                            value += gammaz / (delta_z)
                        end
                    else
                        if BC2 == 1
                            value += gammaz / (delta_z) + gammaz / (delta_z/2)
                        end
                        if BC2 == 2
                            value += gammaz / (delta_z)
                        end
                    end
                end
                
                # X-direction neighbors
                x_neighbors = 0
                if i > 1 && i < nx
                    x_neighbors = 2
                elseif i == 1 || i == nx
                    x_neighbors = 1
                end
                value += x_neighbors * gammax
                
                # Y-direction neighbors
                y_neighbors = 0
                if j > 1 && j < ny
                    y_neighbors = 2
                elseif j == 1 || j == ny
                    y_neighbors = 1
                end
                value += y_neighbors * gammay
                
                push!(I, center_idx); push!(J, center_idx)
                push!(V, value + S)
                
                # Add connections
                if i > 1
                    push!(I, center_idx); push!(J, idx(i-1, j, k)); push!(V, -gammax/delta_x)
                end
                if i < nx
                    push!(I, center_idx); push!(J, idx(i+1, j, k)); push!(V, -gammax/delta_x)
                end
                if j > 1
                    push!(I, center_idx); push!(J, idx(i, j-1, k)); push!(V, -gammay/delta_y)
                end
                if j < ny
                    push!(I, center_idx); push!(J, idx(i, j+1, k)); push!(V, -gammay/delta_y)
                end
                if k > 1
                    push!(I, center_idx); push!(J, idx(i, j, k-1)); push!(V, -gammaz/delta_z)
                end
                if k < nz
                    push!(I, center_idx); push!(J, idx(i, j, k+1)); push!(V, -gammaz/delta_z)
                end
            end
        end
    end
    
    return sparse(I, J, V)
end

function FVM(sp1, sp2, b, nx, ny, nz, temperature_array, Temp=1.0, S=0.0)

    fill!(b, 0.0)
    fill!(temperature_array, 0.0)

    gammaz = 1.0
    delta_z = 1.0
    BC1 = 1
    BC2 = 1
    
    # Helper function to convert 3D index to 1D
    function idx(i, j, k)
        return i + (j-1)*nx + (k-1)*nx*ny
    end
    
    # Only calculate boundary conditions
    for j in 1:ny
        for i in 1:nx
            # Get boundary values
            bottom_val = sp1[i, j] ? Temp : 0.0
            top_val = sp2[i, j] ? Temp : 0.0
            avg_val = (bottom_val + top_val) / 2.0
            
            # Fill all z-levels with average
            for k in 1:nz
                temperature_array[idx(i, j, k)] = avg_val
            end

            # Lower boundary (k = 1)
            if BC1 == 1
                center_idx = idx(i, j, 1)
                b[center_idx] = gammaz * Temp * sp1[i, j] / (delta_z/2)
            end
            
            # Upper boundary (k = nz)
            if BC2 == 1
                center_idx = idx(i, j, nz)
                b[center_idx] = gammaz * Temp * sp2[i, j] / (delta_z/2)
            end
        end
    end
    
    return
end

function find_optimal_threshold(sampled_T, target_pixels, tolerance=0.01)
    # Initialize bounds
    low = 0.0
    high = 1.0
    
    # Helper function to compute pixel count for a threshold
    function count_pixels(threshold)
        mask = sampled_T .>= threshold
        return sum(vec(mask))
    end
    
    # Bisection search
    for _ in 1:100  # Maximum iterations
        mid = (low + high) / 2
        current_count = count_pixels(mid)
        
        # Check if we're close enough
        if abs(current_count - target_pixels) / target_pixels < tolerance
            return mid, current_count
        end
        
        # Adjust bounds
        if current_count > target_pixels
            low = mid
        else
            high = mid
        end
    end
    
    # Return best found value
    final_mid = (low + high) / 2
    return final_mid, count_pixels(final_mid)
end



function create_mapping_groups(mapping_array, max_num_matches)
    groups = Dict{Int, Vector{Int}}()
    group_counter = 1
    
    # Create a mapping from component numbers to their group
    component_to_group = Dict{Int, Int}()
    
    for match1 in 1:max_num_matches
        components = mapping_array[:, match1]
        
        # If any component is 0, create a separate group
        if any(components .== 0)
            groups[group_counter] = [match1]
            for (row_idx, component) in enumerate(components)
                if component != 0
                    # Store component with its row position
                    component_to_group[row_idx * 10000 + component] = group_counter
                end
            end
            group_counter += 1
            continue
        end
        
        # Find existing group for these components
        existing_groups = Set{Int}()
        for (row_idx, component) in enumerate(components)
            # Use row_idx in the key to ensure row-specific matching
            component_key = row_idx * 10000 + component
            if haskey(component_to_group, component_key)
                push!(existing_groups, component_to_group[component_key])
            end
        end
        
        if isempty(existing_groups)
            # Create new group
            groups[group_counter] = [match1]
            for (row_idx, component) in enumerate(components)
                component_to_group[row_idx * 10000 + component] = group_counter
            end
            group_counter += 1
        else
            # Add to existing group
            target_group = minimum(existing_groups)
            push!(groups[target_group], match1)
            
            # Merge groups if necessary
            for group in existing_groups
                if group != target_group
                    append!(groups[target_group], groups[group])
                    delete!(groups, group)
                end
            end
            
            # Update component mappings
            for (row_idx, component) in enumerate(components)
                component_to_group[row_idx * 10000 + component] = target_group
            end
        end
    end
    
    return groups
end

function diffusion_smoothing(path::String, slice1::Int64, slice2::Int64, xloc::Int64, yloc::Int64, height::Int64, nx::Int64, ny::Int64, nz::Int64, step_per_layer::Int64, rank::Int64, a_sparse::SparseMatrixCSC{Float64,Int64},P::IncompleteLU.ILUFactorization{Float64, Int64},sp1_binary::Array{Bool,2},sp2_binary::Array{Bool,2},b::Array{Float64,1},temperature_array::Array{Float64,1})
    @show rank,slice1
    
    step = 1  # Show every  point
    
    
    filename = string(path,"sparse_arrays_",slice1,"_",xloc,"_",yloc,"_",height,".jld2")
    @load filename sparse_arrays sparse_arrays_components
    sparse_arrays1::Dict{Tuple{Int64,Int64}, SparseMatrixCSC{Float64, Int64}} = sparse_arrays
    sparse_arrays_components1::Dict{Int64,Int64} = sparse_arrays_components

    filename = string(path,"sparse_arrays_",slice2,"_",xloc,"_",yloc,"_",height,".jld2")
    @load filename sparse_arrays sparse_arrays_components
    sparse_arrays2::Dict{Tuple{Int64,Int64}, SparseMatrixCSC{Float64, Int64}} = sparse_arrays
    sparse_arrays_components2::Dict{Int64,Int64} = sparse_arrays_components
    
    # Create a dictionary of unique keys with tuples of values
    combined_components = Dict{Int64, Tuple{Int64, Int64}}()
    for key in union(keys(sparse_arrays_components1), keys(sparse_arrays_components2))
        value1 = get(sparse_arrays_components1, key, 0)
        value2 = get(sparse_arrays_components2, key, 0)
        combined_components[key] = (value1, value2)
    end

    
    # After creating combined_components
    component_mapping = Dict{Int64, Dict{Int64, Tuple{Int64,Int64}}}()
    num_matches = Dict{Int64, Tuple{Int64, Int64}}()
    # Loop through each component pair in combined_components
    fill!(sp1_binary, false)
    fill!(sp2_binary, false)

    result_dict = Dict{Int64, Dict{Int64, SparseMatrixCSC{Float64, Int64}}}()
    for (cell_number, (key, (value1, value2))) in enumerate(combined_components)
  
        @show rank,cell_number

        component_mapping_matches = Dict{Int64, Tuple{Int64,Int64}}()
        
        if value1 > 0 && value2 > 0
            counter = 0
            match = false
            
            for i in 1:value1
                sp1 = sparse_arrays1[key,i]
                rows1, cols1, _ = findnz(sp1)
                fill!(sp1_binary, false)
                for (row, col) in zip(rows1, cols1)
                    sp1_binary[row, col] = true
                end
                
                for j in 1:value2
                    sp2 = sparse_arrays2[key,j]
                    rows2, cols2, _ = findnz(sp2)
                    fill!(sp2_binary, false)
                    for (row, col) in zip(rows2, cols2)
                        sp2_binary[row, col] = true
                    end
                    # Calculate overlap
                    overlap = sum(sp1_binary .& sp2_binary)
                    if overlap > 0
                        match = true
                        counter += 1
                        component_mapping_matches[counter] = (i,j)
                    end
                end
                if match == false
                    counter += 1
                    component_mapping_matches[counter] = (i,0)
                else
                    match = false
                end
            end
           
            for j in 1:value2
                
                match = false
                for (key,value) in component_mapping_matches
                    if value[2] == j
                        match = true
                    end
                end
                if match == false
                    counter += 1
                    component_mapping_matches[counter] = (0,j)
                else
                    match = false
                end
            end
        end
            
        
        if value1 > 0 && value2 == 0
            counter = 0
            for i in 1:value1
                    counter += 1
                    component_mapping_matches[counter] = (i,0)
                
            end
        end
        if value2 > 0 && value1 == 0
            counter = 0
            for j in 1:value2
                    counter += 1
                    component_mapping_matches[counter] = (0,j)

            end
        end
        num_matches[key] =  (counter, cell_number)
        
        fill!(sp1_binary, false)
        fill!(sp2_binary, false)
        for i in 1:value1 
            
            sp1 = sparse_arrays1[key,i]
            rows1, cols1, _ = findnz(sp1)
            # Fill binary masks for sp1 and sp2
            for (row, col) in zip(rows1, cols1)
                sp1_binary[row, col] = true
            end
        end
        for i in 1:value2
            
            sp2 = sparse_arrays2[key,i]
            rows2, cols2, _ = findnz(sp2)
            for (row, col) in zip(rows2, cols2)
                sp2_binary[row, col] = true
            end
        end
        Temp = 10.0
        S = 0.0
        FVM(sp1_binary, sp2_binary, b, nx, ny, nz, temperature_array, Temp, S)
        
        
        
        cg!(temperature_array, a_sparse, b; maxiter=400, Pl=P)
        T_3d_1 = reshape(temperature_array, nx, ny, nz)
        
       
        
  
        layer_dict = Dict{Int64, SparseMatrixCSC{Float64, Int64}}()
        counter = 1
        # layers = [1,7,13,19,25]
        layer = 7
        # for i in layers
            
            layer_T = T_3d_1[1:step:end, 1:step:end, layer]
            # Create sparse array with threshold
            mask = layer_T .> 0.05
            indices = findall(mask)
            rows = getindex.(indices, 1)
            cols = getindex.(indices, 2)
            vals = layer_T[mask]
            layer_dict[slice1*5+counter] = sparse(rows, cols, vals, nx, ny)
            counter += 1
        # end
        result_dict[key] = layer_dict
        component_mapping[key] = component_mapping_matches
        
      
    end

    group_dict = Dict{Int64, Tuple{Dict{Int64, Vector{Int64}}, Matrix{Int64}}}()
    # Loop through num_matches to find all mappings
    for (key, (max_num_matches, cell_number)) in num_matches
        
        mapping_array = Array{Int64}(undef, 2, max_num_matches)
        for match in 1:max_num_matches
           
            # Read the component mapping for this key and match into an array
            mapping_array[:,match] .= component_mapping[key][match]
        end
        groups = create_mapping_groups(mapping_array, max_num_matches)
        if length(groups) > 1
            println("\nkey $key:")
            for (group_id, matches) in groups
            
                println("Group $group_id:")
                for match in matches
                    println("  Match $match: ", mapping_array[:, match])
                end
            end
        end
        group_dict[key] = (groups, mapping_array)
    end
    # After the main loop that creates result_dict, add:
    output_filename = string(path, "result_dict_", slice1, "_", xloc, "_", yloc, "_", height, ".jld2")
    @save output_filename result_dict group_dict component_mapping combined_components num_matches
    
    println("Finished processing slice $slice1")
  

    return 
end

function job_queue(path::String,start_slice::Int64,end_slice::Int64,xloc::Int64,yloc::Int64,height::Int64,nx::Int64,ny::Int64,nz::Int64,step_per_layer::Int64,comm::MPI.Comm,rank::Int64,world_size::Int64,a_sparse::SparseMatrixCSC{Float64,Int64},P::IncompleteLU.ILUFactorization{Float64, Int64},sp1_binary::Array{Bool,2},sp2_binary::Array{Bool,2},b::Array{Float64,1},temperature_array::Array{Float64,1})

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
                    
                    diffusion_smoothing(path,slice1,slice2,xloc,yloc,height,nx,ny,nz,step_per_layer,rank,a_sparse,P,sp1_binary,sp2_binary,b,temperature_array)

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
    
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    if world_size <= 1
        println("World size must be greater than 1")
        return
    end
    a_sparse = calculate_FVM_matrix(nx, ny, nz)

    P = ilu(a_sparse, τ = 0.1)
    @show typeof(P)
    sp1_binary = fill(false, nx, ny)
    sp2_binary = fill(false, nx, ny)
    b = zeros(Float64, nx * ny * nz)
    temperature_array = zeros(Float64, nx * ny * nz)
    job_queue(path,start_slice,end_slice,xloc,yloc,height,nx,ny,nz,step_per_layer,comm,rank,world_size,a_sparse,P,sp1_binary,sp2_binary,b,temperature_array)


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
# slices = 1
end_slice = start_slice + slices

step_per_layer = 5
nx, ny, nz = 501, 501, 25



global_setup(path,start_slice,end_slice,xloc,yloc,height,nx,ny,nz,step_per_layer)