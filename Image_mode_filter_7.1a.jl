using MPI
using CSV
using Tables

function process_3x3_kernel!(matrix, slice, marked_for_removal, height, width)
    neighborhood = zeros(Int,3,3)
    count = 1
    
    while count > 0
        count = 0
        # Process interior points
        for i in 2:height-1
            for j in 2:width-1
                if !marked_for_removal[i,j]
                    continue
                end
                
                @views neighborhood .= matrix[i-1:i+1, j-1:j+1]
                center = matrix[i, j]
                
                counts = Dict{Int, Int}()
                for val in neighborhood
                    counts[val] = get(counts, val, 0) + 1
                end
                
                try
                    delete!(counts, center)
                catch e
                    @show e, center, counts, neighborhood
                end
                
                if !isempty(counts)
                    mode_count, mode_val = findmax(counts)
                    if mode_count > 0
                        marked_for_removal[i,j] = false
                        matrix[i, j] = mode_val
                        count += 1
                    end
                else
                    marked_for_removal[i,j] = false
                end
            end
        end

        # Process edges
        # Top edge (i=1)
        for j in 2:width-1
            if marked_for_removal[1,j]
                neighbors = [matrix[1,j-1], matrix[1,j+1], matrix[2,j], matrix[2,j-1], matrix[2,j+1]]
                counts = Dict{Int, Int}()
                for val in neighbors
                    counts[val] = get(counts, val, 0) + 1
                end
                if !isempty(counts)
                    _, mode_val = findmax(counts)
                    marked_for_removal[1,j] = false
                    matrix[1,j] = mode_val
                    count += 1
                end
            end
        end

        # Bottom edge
        for j in 2:width-1
            if marked_for_removal[height,j]
                neighbors = [matrix[height,j-1], matrix[height,j+1], matrix[height-1,j], matrix[height-1,j-1], matrix[height-1,j+1]]
                counts = Dict{Int, Int}()
                for val in neighbors
                    counts[val] = get(counts, val, 0) + 1
                end
                if !isempty(counts)
                    _, mode_val = findmax(counts)
                    marked_for_removal[height,j] = false
                    matrix[height,j] = mode_val
                    count += 1
                end
            end
        end

        # Left edge
        for i in 2:height-1
            if marked_for_removal[i,1]
                neighbors = [matrix[i-1,1], matrix[i+1,1], matrix[i,2], matrix[i-1,2], matrix[i+1,2]]
                counts = Dict{Int, Int}()
                for val in neighbors
                    counts[val] = get(counts, val, 0) + 1
                end
                if !isempty(counts)
                    _, mode_val = findmax(counts)
                    marked_for_removal[i,1] = false
                    matrix[i,1] = mode_val
                    count += 1
                end
            end
        end

        # Right edge
        for i in 2:height-1
            if marked_for_removal[i,width]
                neighbors = [matrix[i-1,width], matrix[i+1,width], matrix[i,width-1], matrix[i-1,width-1], matrix[i+1,width-1]]
                counts = Dict{Int, Int}()
                for val in neighbors
                    counts[val] = get(counts, val, 0) + 1
                end
                if !isempty(counts)
                    _, mode_val = findmax(counts)
                    marked_for_removal[i,width] = false
                    matrix[i,width] = mode_val
                    count += 1
                end
            end
        end

        # Corners
        # Process corners only if they're marked for removal
        corners = [(1,1), (1,width), (height,1), (height,width)]
        for (i,j) in corners
            if marked_for_removal[i,j]
                neighbors = if (i,j) == (1,1)
                    [matrix[1,2], matrix[2,1]]
                elseif (i,j) == (1,width)
                    [matrix[1,width-1], matrix[2,width]]
                elseif (i,j) == (height,1)
                    [matrix[height-1,1], matrix[height,2]]
                else # (height,width)
                    [matrix[height-1,width], matrix[height,width-1]]
                end
                
                counts = Dict{Int, Int}()
                for val in neighbors
                    counts[val] = get(counts, val, 0) + 1
                end
                if !isempty(counts)
                    _, mode_val = findmax(counts)
                    marked_for_removal[i,j] = false
                    matrix[i,j] = mode_val
                    count += 1
                end
            end
        end
        if slice == 321
            @show slice,"active",count,sum(marked_for_removal)
        end
    end
end

function mark_and_process_matrix!(matrix, marked_for_removal, frame_length1, frame_length2, total_frames, neighborhood, neighbormask, slice, iterations=50)
    for idx in 1:iterations
        # Initial marking phase
        for i in 2:frame_length1-1
            for j in 2:frame_length2-1
                @views neighborhood .= matrix[i-1:i+1, j-1:j+1]
                center = matrix[i, j]
                
                counts = Dict{Int, Int}()
                for val in neighborhood
                    counts[val] = get(counts, val, 0) + 1
                end
                
                mode_count, mode_val = findmax(counts)
                
                if center != mode_val
                    marked_for_removal[i,j] = true
                elseif mode_count < 4
                    marked_for_removal[i,j] = true
                elseif mode_count == 4
                    neighbormask .= neighborhood .== center
                    if (neighbormask[2,1] + neighbormask[1,2] + neighbormask[2,3] + neighbormask[3,2]) <= 1
                        marked_for_removal[i,j] = true
                    end
                end
            end
        end

        # Process edges
        mark_edges!(matrix, marked_for_removal, frame_length1, frame_length2, total_frames)
        if slice == 321
            @show slice,"mode filter",sum(marked_for_removal)
        end
        # Process with 3x3 kernel
        process_3x3_kernel!(matrix, slice, marked_for_removal, frame_length1, frame_length2)
        # if sum(marked_for_removal) == 0
        #     @show slice,"mode filter",sum(marked_for_removal)
        # end
    end
end

function mark_edges!(matrix, marked_for_removal, frame_length1, frame_length2, total_frames)
    # Top edge (i=1)
    for j in 2:frame_length2-1
        center = matrix[1, j]
        if center != matrix[1, j-1] || center != matrix[1, j+1] || center != matrix[2, j]
            marked_for_removal[1,j] = true
        end
    end

    # Bottom edge (i=height)
    for j in 2:frame_length2-1
        center = matrix[frame_length1, j]
        if center != matrix[frame_length1, j-1] || center != matrix[frame_length1, j+1] || center != matrix[frame_length1-1, j]
            marked_for_removal[frame_length1,j] = true
        end
    end

    # Left edge (j=1)
    for i in 2:frame_length1-1
        center = matrix[i, 1]
        if center != matrix[i-1, 1] || center != matrix[i+1, 1] || center != matrix[i, 2]
            marked_for_removal[i,1] = true
        end
    end

    # Right edge (j=width)
    for i in 2:frame_length1-1
        center = matrix[i, frame_length2]
        if center != matrix[i-1, frame_length2] || center != matrix[i+1, frame_length2] || center != matrix[i, frame_length2-1]
            marked_for_removal[i,frame_length2] = true
        end
    end

    # Corner cases
    # Top-left corner
    if matrix[1,1] != matrix[1,2] || matrix[1,1] != matrix[2,1]
        marked_for_removal[1,1] = true
    end
    # Top-right corner
    if matrix[1,frame_length2] != matrix[1,frame_length2-1] || matrix[1,frame_length2] != matrix[2,frame_length2]
        marked_for_removal[1,total_frames] = true
    end
    # Bottom-left corner
    if matrix[frame_length1,1] != matrix[frame_length1-1,1] || matrix[frame_length1,1] != matrix[frame_length1,2]
        marked_for_removal[frame_length1,1] = true
    end
    # Bottom-right corner
    if matrix[frame_length1,frame_length2] != matrix[frame_length1-1,frame_length2] || matrix[frame_length1,frame_length2] != matrix[frame_length1,frame_length2-1]
        marked_for_removal[frame_length1,frame_length2] = true
    end
end

function process_neuron_data_xyz(path, slice, extension, comm, rank, world_size, xcounter, ycounter, zcounter, counter, location, startframe, endframe, xval, yval, height, slices, xy_elements,win_all_frames,xyz)
    
    # @show rank, slice, slices,startframe, xval, yval,xy_elements
    frame_height = xy_elements+1
    frame_width = xy_elements+1
    total_frames = slices  + 1
    
    # @show frame_height,frame_width,total_frames
    if xyz == 1
        frame_length1 = frame_height
        frame_length2 = frame_width
        frame_length3 = total_frames
        
    elseif xyz == 2
        frame_length1 = frame_height
        frame_length2 = total_frames
        frame_length3 = frame_width
      
    elseif xyz == 3
        frame_length1 = frame_width
        frame_length2 = total_frames
        frame_length3 = frame_height
        
    end
    matrix = zeros(Int64,frame_length1,frame_length2)
    MPI.Win_lock(win_all_frames[slice];rank=0,type=:exclusive)
    MPI.Get!(matrix, win_all_frames[slice];rank=0)
    MPI.Win_unlock(win_all_frames[slice];rank=0)

    neighborhood = zeros(Int,3,3)
    neighbormask = zeros(Bool,3,3)
    marked_for_removal = zeros(Bool, size(matrix))
    if slice == 321
        @show slice,"start",sum(marked_for_removal)
    end
    
    mark_and_process_matrix!(matrix, marked_for_removal, frame_length1, frame_length2, total_frames, neighborhood, neighbormask, slice)
    if slice == 321
        @show slice,"after mode 1",sum(marked_for_removal)
    end
    

    # 11x11 kernel check
    
    for i in 1:frame_length1
        for j in 1:frame_length2
            i_start = max(1, i-5)
            i_end = min(frame_length1, i+5)
            j_start = max(1, j-5)
            j_end = min(frame_length2, j+5)
            
            kernel = matrix[i_start:i_end, j_start:j_end]
            edge_values = Set{Int}()
            
            if i-5 > 0
                union!(edge_values, kernel[1,:])
            end
            if i+5 < frame_length1
                union!(edge_values, kernel[end,:])
            end
            if j-5 > 0
                union!(edge_values, kernel[:,1])
            end
            if j+5 < frame_length2
                union!(edge_values, kernel[:,end])
            end
            
            center_val = matrix[i,j]
            if center_val ∉ edge_values && center_val != 0
                marked_for_removal[i,j] = true
            end

            if i > 2 && i < frame_length1-1
                if center_val == matrix[i-1,j] && center_val == matrix[i+1,j] && center_val != matrix[i-2,j] && center_val != matrix[i+2,j]
                    marked_for_removal[i,j] = true
                    marked_for_removal[i-1,j] = true
                    marked_for_removal[i+1,j] = true
                end
            end
            if j > 2 && j < frame_length2-1
                if center_val == matrix[i,j-1] && center_val == matrix[i,j+1] && center_val != matrix[i,j-2] && center_val != matrix[i,j+2]
                    marked_for_removal[i,j] = true
                    marked_for_removal[i,j-1] = true
                    marked_for_removal[i,j+1] = true
                end
            end
        end
    end
    if slice == 321
        @show slice,"square mark",sum(marked_for_removal)
    end
    # Process with 3x3 kernel
    process_3x3_kernel!(matrix, slice, marked_for_removal, frame_length1, frame_length2)
    if slice == 321
        @show slice,"after square",sum(marked_for_removal)
    end

    mark_and_process_matrix!(matrix, marked_for_removal, frame_length1, frame_length2, total_frames, neighborhood, neighbormask, slice)
    if slice == 321
        @show slice,"after mode 2",sum(marked_for_removal)
    end
    
    
    MPI.Win_lock(win_all_frames[slice];rank=0,type=:exclusive)
    MPI.Put!(MPI.Buffer_send(matrix), win_all_frames[slice];rank=0)
    MPI.Win_unlock(win_all_frames[slice];rank=0)

        
   
    return
end


function job_queue(path,extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,xval,yval,height,slices,xy_elements,win_all_frames,xyz)

    nworkers = world_size - 1

    root = 0

    MPI.Barrier(comm)
    T = eltype(slices)
    N = slices+1
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
                    print("Worker $rank: Received number $(recv_mesg[1]) from root\n")

                    process_neuron_data_xyz(path,recv_mesg[1],extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,xval,yval,height,slices,xy_elements,win_all_frames,xyz)

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

function global_setup(sim_number::Int64, path::String)
    MPI.Init()
    root = 0
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    if world_size <= 1
        println("World size must be greater than 1")
        return
    end
    path_variables = Vector{Int64}(undef,6)
    location = zeros(Int64,3,2)

    
    if rank == 0
        filename = string(@__DIR__,"/Simulation_settings_4.0.csv")
        settings = CSV.File(filename) |> Tables.matrix
        
             
        xval::Int64 = settings[sim_number,2]
        yval::Int64 = settings[sim_number,3]
        startframe::Int64 = settings[sim_number,4]
        height::Int64 = settings[sim_number,5]
        
        
        
        location[1,1:2] .= settings[sim_number,6:7]
        location[2,1:2] .= settings[sim_number,8:9]
        location[3,1:2] .= settings[sim_number,10:11]
        z_filled::Int64 = settings[sim_number,12]
        if z_filled == 1
            slices = Int64(height*25*5)
        else
            slices = Int64(height*25)
        end

        for i in 1:3
            if location[i,1] > 0
                location[i,1] = -location[i,1]
            end
            if location[i,2] < 0
                println("Error: plus location must be positive")
                return
            end 
        end

        path_variables .= slices,startframe,xval,yval,height,z_filled
        
       
    end
    MPI.Barrier(comm)

    MPI.Bcast!(location, comm; root=0)

    cubes = (location[1,2]-location[1,1]+1)*(location[2,2]-location[2,1]+1)*(location[3,2]-location[3,1]+1)
    
    cube_face_match_collection = Vector{Vector{Int32}}(undef,cubes)
    MPI.Barrier(comm)

    MPI.Bcast!(path_variables, comm; root=0)
    slices,startframe,xval,yval,height,z_filled = path_variables
    xy_elements = height*125
    MPI.Barrier(comm)


    
    null_array = Vector{Int32}(undef,1)
    counter = 0
    for xcounter in location[1,1]:location[1,2]
        for ycounter in location[2,1]:location[2,2]
            for zcounter in location[3,1]:location[3,2]
                MPI.Barrier(comm)
                counter += 1
                current_startframe = startframe + zcounter*slices
                current_xval = xval + xcounter*xy_elements
                current_yval = yval + ycounter*xy_elements
                total_frames = slices  + 1
                frame_height = xy_elements+1
                frame_width = xy_elements+1
                if rank == 0
                    all_frames = zeros(Int64, frame_width,frame_height,total_frames)
                            
                    matrix = Array{Int64,2}(undef,frame_width,frame_height)
                
                    @show xcounter,ycounter,zcounter
                    @show current_startframe,current_xval,current_yval  
                    
                    # Initialize 3D array to store all frames
                    
                    @show size(matrix),size(all_frames)
                    
                    for slice in 1:total_frames
                        if z_filled == 0
                            input_file = string(path,"frame_", current_startframe+slice-1, "_", xval, "_", yval, "_",height,".csv")
                        else
                            input_file = string(path,"zframe_", current_startframe+slice-1, "_", xval, "_", yval, "_",height,".csv")
                        end
                        # println("Starting to read file: $input_file")
                        # @show input_file
                        # Read frame data
                        
                        data = parse.(Int, replace.(split(readline(input_file), ","), r"^x" => ""))
                        width = length(data)
                        if width != frame_width
                            println("Error: Width of data does not match frame width")
                            return
                        end
                     
                        for (i,line) in enumerate(eachline(input_file))
                            if i == 1
                                continue
                            end
                            # @show i-1,(i-2)*frame_width+1:(i-1)*frame_width,size(parse.(Int, split(line, ",")))
                            # println("Processing line $i")
                            vals = parse.(Int, split(line, ","))
                            matrix[i-1,:] .= vals
                            # idx += frame_width
                        end
                        # println(string("Finished reading file ",slice))

                        
                        all_frames[:,:,slice] .= matrix
                     
                    end
                end
                if z_filled == 0
                    end_iter = 1
                    end_xyz = [1]
                else
                    end_iter = 10
                    end_xyz = [1,2,3,1]
                end
                for iter in 1:end_iter
                    for xyz in end_xyz
                        
                        frame_length = xyz == 1 ? total_frames : xyz == 2 ? frame_width : frame_height   
                        if rank == 0
                            all_frames_vector = Vector{Array{Int64,2}}(undef,frame_length)
                            win_all_frames =  Vector{MPI.Win}(undef,frame_length)
                            
                            
                            for j in 1:frame_length
                              
                                if xyz == 1 
                                    all_frames_vector[j] = all_frames[:,:,j]
                                elseif xyz == 2
                                    all_frames_vector[j] = all_frames[:,j,:]
                                elseif xyz == 3
                                    all_frames_vector[j] = all_frames[j,:,:]
                                end
                                win_all_frames[j] = MPI.Win_create(all_frames_vector[j], comm)
                            end
                        
                        else
                            win_all_frames =  Vector{MPI.Win}(undef,frame_length)
                            for j in 1:frame_length
                                win_all_frames[j] = MPI.Win_create(null_array, comm)   
                            end
                           
                        end
                        MPI.Barrier(comm)
                        extension = string("_",current_startframe,"_",current_xval,"_",current_yval,"_",height)
            
                        endframe = current_startframe+slices
                        
                 
                        
                        job_queue(path,extension,comm,rank,world_size,xcounter,ycounter,zcounter,counter,location,current_startframe,endframe,current_xval,current_yval,height,slices,xy_elements,win_all_frames,xyz)

                        MPI.Barrier(comm)
                        if rank == 0
                            @show xcounter,ycounter,zcounter
                            @show current_startframe,current_xval,current_yval  
                            total_frames = slices  + 1
                            frame_height = xy_elements+1
                            frame_width = xy_elements+1
                            @show frame_height, frame_width, total_frames
                            
                            for j in 1:frame_length
                                # for i in 1:frame_width
                                #     for slice in 1:total_frames
                                #         all_frames_temp[i,slice] = all_frames[i,j,slice]
                                #     end
                                # end
                                if xyz == 1
                                    all_frames[:,:,j] .= all_frames_vector[j]
                                elseif xyz == 2
                                    all_frames[:,j,:] .= all_frames_vector[j]
                                elseif xyz == 3
                                    all_frames[j,:,:] .= all_frames_vector[j]
                                end
                                
                            end
                        end
                        MPI.Barrier(comm)
                        for j in 1:frame_length
                            MPI.free(win_all_frames[j])
                        end
                    end
                end
                MPI.Barrier(comm)
                if rank == 0
                    result = Array{Int64,2}(undef,frame_width,frame_height)
                    for slice in 1:total_frames
                        if z_filled == 0
                            output_file = string(path,"sframe_", current_startframe+slice-1, "_", xval, "_", yval, "_",height,".csv")
                        else
                            output_file = string(path,"szframe_", current_startframe+slice-1, "_", xval, "_", yval, "_",height,".csv")
                        end
                        # println("Starting to read file: $input_file")
                        # @show input_file
                        # Read frame data
                        
                        
                        result .= all_frames[:,:,slice]
                        # @show slice,result[245:255,245:255]
                        # Write output
                        open(output_file, "w") do f
                            println(f, join(0:frame_width-1, ","))
                            for i in 1:frame_height
                                println(f, join(result[i,:], ","))
                            end
                        end
                    end
                    
                end
            
            
         
                
            
            
            end
        end
    end
    
    
end


path = string(@__DIR__, "/../Neuron_transport_data/")

for sim_number in 20:20:20
    global_setup(sim_number,path)
end

