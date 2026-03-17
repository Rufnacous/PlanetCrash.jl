module PlanetCrash

    using Plots, DifferentialEquations, LinearAlgebra, Distances


    function test()

        N = 30
        initcond = zeros(5N);
        positions = @view initcond[begin:2N];
        velocities = @view initcond[(2N+1):(4N)];
        temps = @view initcond[(4N+1):end];

        temps .= 1 .+ 0.1rand(N)


        # positions[3] = 1
        # positions[1] = -positions[3]

        positions .= 5rand(2N);
        # positions[begin:2:Integer(round(N/2))] .+= 20;

        # velocities[12:2:end] .+= 20;

        tspan = [0,20];
        
        # p = (N, 1, 0.1, 0.2, 0.0003, 100, 0.0);
        p = (N, 1, 0.1, 0.2, 0.0003, 100, 0.01, 10);
        prob = ODEProblem(v_cauchy, initcond, tspan, p)
        sol = solve(prob, TRBDF2(autodiff=false))


        

        default(show=true);

        ts = LinRange(0, tspan[end], 1+Integer(round(10*tspan[end])));
        xys = zeros(N,2, length(ts))
        temp_s = zeros(N, length(ts))
        for (ti, t) in enumerate(ts)
            sol_ti = sol(t);
            xys[:,:,ti] .= reshape(sol_ti[1:2N], (2,N))';
            temp_s[:,ti] .= sol_ti[(4N+1):5N];
        end
        # display(xys)
        # display(temp_s)

        plot([],[],label="", aspect_ratio=:equal);
        for ni in 1:N
            plot!(xys[ni,1,:], xys[ni,2,:], label="", linez=temp_s[ni,:], c=:turbo, linewidth=2);
        end


        print(">");
        if readline() == "q"
            return
        end

        initcond2 = sol(tspan[end])[1:2N]
        initcond2[1:2:end] .-= initcond2[1]
        initcond2[2:2:end] .-= initcond2[2]
        Nold = N
        N = 2N
        initcond2 = [initcond2..., initcond2..., (1 .+ 0.1rand(N))..., zeros(3N)...];
        
        positions = @view initcond2[begin:2N];
        velocities = @view initcond2[(2N+1):end];
        
        sep = 15
        positions[1:2:(2Nold)] .-= sep
        positions[(2Nold+1):2:end] .+= sep


        
        velocities[1:2:(2Nold)] .= 2.0
        velocities[(2Nold+1):2:end] .= -2.0
        
        velocities[2:2:(2Nold)] .= 0.07
        velocities[(2Nold+2):2:end] .= -0.07

        tspan = [0,60];
        p = (N, 1, 0.1, 0.2, 0.00003, 0.005, 0.03, 40);  # test 2

        checkpoints = [perc*tspan[end] for perc in 0.01:0.01:1.0]

        prob = ODEProblem(v_cauchy, initcond2, tspan, p, tstops = checkpoints)

        condition(u, t, integrator) = any(t .== checkpoints);
        affect!(integrator) = println("T = $(integrator.t)");
        cb_checkpoints = DiscreteCallback(condition, affect!);
        
        sol = solve(prob, TRBDF2(autodiff=false), callback=cb_checkpoints)



        default(show=false);

        FPS = 5
        ts = LinRange(0, tspan[end], 1+Integer(round(FPS*tspan[end])));

        mintemp = 0 #min(0, minimum([minimum(sol(t)[(4N+1):5N]) for t in ts]))
        maxtemp = max(1, maximum([maximum(sol(t)[(4N+1):5N]) for t in ts]))

        # ENV["GKSWSTYPE"] = "" #100 #""
        println("Plotting")
        anim = @animate for (ti,t) in enumerate(ts)
            sol_ti = sol(t)[1:5N];

            xs, ys = sol_ti[1:2:2N], sol_ti[2:2:2N]
            temps = sol_ti[(4N+1):5N]

            xs .-= sum(xs)/N
            ys .-= sum(ys)/N

            lim = max(4, 1.5max(maximum(abs.(xs)),maximum(abs.(ys))))

            scatter(xs, ys, 
                title="$(Integer(round(10t))/10)", aspect_ratio=:equal,
                xlims=(-lim,lim), ylims=(-lim,lim), framestyle=:box,
                label="", 
                zcolor = temps, color=:turbo, clims=(mintemp,maxtemp),
                markerstrokewidth = 0,
                # zcolor=[ifelse(i<=0.5N, -1, 1) for i in 1:N], cbar=nothing, color=:bam, clims=(-2,2)
                )

        end

        gif(anim, "planet_crash.gif", fps=15)

        # plot(sol.t,label="",ylabel="t",xlabel="ti",framestyle=:box)
        # savefig("tsteps.png")


        # plot(sol, idxs=1:2:2N)



    end



    function v_cauchy(du, u, p, t)
        N = p[1];
        m = p[2];
        k = p[3];
        G = p[4];
        c = p[5];
        P = p[6];
        E = p[7];
        dc = p[8];

        vel = @view u[(2N+1):(4N)]
        temp = @view u[(4N+1):(5N)]
        du[begin:2N] .= vel;

        # dp = pairwise(Euclidean);

        x = @view u[1:2:2N];
        y = @view u[2:2:2N];

        vx = @view vel[1:2:end];
        vy = @view vel[2:2:end];

        dx = x .- reshape(x, (1,N));
        dy = y .- reshape(y, (1,N));

        dvx = vx .- reshape(vx, (1,N));
        dvy = vy .- reshape(vy, (1,N));

        dt = temp .- reshape(temp, (1,N));


        dp = cat(dx, dy, dims=3);
        dvp = cat(dvx, dvy, dims=3);

        dr2 = sum(dp .^ 2, dims=3);
        dr  = sqrt.(dr2);
        dr3 = dr2 .* dr;
        dv  = sqrt.(sum(dvp .^ 2,dims=3));

        inv_dr = 1 ./ dr
        inv_dr2 = 1 ./ dr2
        inv_dr3 = 1 ./ dr3
        inv_dv = 1 ./ dv
        
        setindex!.(Ref(inv_dr), 0.0, 1:N, 1:N)
        setindex!.(Ref(inv_dr2), 0.0, 1:N, 1:N)
        setindex!.(Ref(inv_dr3), 0.0, 1:N, 1:N)
        setindex!.(Ref(inv_dv), 0.0, 1:N, 1:N)

        dp = dp .* inv_dr # normalized dp for unit vectors
        dvp = dvp .* inv_dv
        dvp[.!(isfinite.(dvp))] .= 0

        
        
        a = -(( ((G*m) .* inv_dr2)) .- ((k/m) .* inv_dr3) )
        particle_force = reshape( reshape(sum(a .* dp,dims=2),(N,2))', 2N );
        
        damp =  sum((c / m) .* (dv .* inv_dr3), dims=1)[:] 
        
        resistive_force = reshape( reshape(sum(damp .* dvp,dims=2),(N,2))', 2N);


        du[(2N+1):(4N)] .= particle_force .- resistive_force;

        du[(4N+1):(5N)] .=  (dc .* damp) + sum( (P/m) .* (dt .* inv_dr), dims=1 )[:] .- (E .* temp);
        

    end

    function interleave(a)
        b = a
        c = Vector(undef, 2*length(a))
        i = 0
        for (x, y) in zip(a, b)
            c[i += 1] = x
            c[i += 1] = y
        end
        return c
    end

end
