module PlanetCrash

    using Plots, DifferentialEquations, LinearAlgebra, Distances, DelimitedFiles

    function init_planet(N, m, k, G; T=100, c=0.0003, P=100, E=0.01, dc=10)

        initcond = zeros(5N);
        initcond[begin:2N] .= 5rand(2N);
        initcond[(4N+1):end] .= 1;

        prob = ODEProblem(v_cauchy, initcond, [0,T], (N, m, k, G, c, P, E, dc))
        sol = solve(prob, TRBDF2(autodiff=false))

        outcond_pos = sol(sol.t[end])[1:2N]
        mean_x = sum(outcond_pos[1:2:end])/N
        mean_y = sum(outcond_pos[2:2:end])/N
        outcond_pos[1:2:end] .-= mean_x
        outcond_pos[2:2:end] .-= mean_y

        writedlm( "./planets/N$(N)_m$(m)_k$(k)_g$(G).csv", outcond_pos, ',' )
    end

    function spin_planet(xys, theta_dot)
        vs = zeros(size(xys));
        rs = sqrt.((xys[1:2:end] .^ 2) .+ (xys[2:2:end] .^ 2));
        vs[1:2:end] .= -xys[2:2:end] .* theta_dot;
        vs[2:2:end] .=  xys[1:2:end] .* theta_dot;
        return vs;
    end

    function test()

        N1 = 30; N2 = 20; N = N1 + N2;
        m = 1; k = 0.1; G = 0.2;
        sep = 5

        planet1 = readdlm( "./planets/N$(N1)_m$(m)_k$(k)_g$(G).csv", ',', Float64, '\n');
        planet2 = readdlm( "./planets/N$(N2)_m$(m)_k$(k)_g$(G).csv", ',', Float64, '\n');
        initcond = zeros(5N);
        initcond[1:(2N1)] .= planet1;
        initcond[1:2:(2N1)] .-= sep;
        initcond[(2N1+1):(2N)] .= planet2;
        initcond[(2N1+1):2:(2N)] .+= sep;

        # initcond[(2N+1):(2N+2N1)] .= spin_planet(planet1, 1.4)
        initcond[(2N+2N1+1):2:(4N)] .= -3.0
        initcond[(2N+2N1+2):2:(4N)] .= -0.5

        tspan = [0,30];
        p = (N, m, k, G, 0.00003, 0.005, 0.03, 40);

        checkpoints = [perc*tspan[end] for perc in 0.01:0.01:1.0]
        condition(u, t, integrator) = any(t .== checkpoints);
        affect!(integrator) = println("T = $(integrator.t)");
        cb_checkpoints = DiscreteCallback(condition, affect!);

        prob = ODEProblem(v_cauchy, initcond, tspan, p, tstops = checkpoints)
        
        sol = solve(prob, TRBDF2(autodiff=false), callback=cb_checkpoints)

        default(show=true);

        FPS = 15
        ts = LinRange(0, tspan[end], 1+Integer(round(FPS*tspan[end])));

        mintemp = minimum([minimum(sol(t)[(4N+1):5N]) for t in ts])
        maxtemp = maximum([maximum(sol(t)[(4N+1):5N]) for t in ts])
        midtemp = 0.5(mintemp + maxtemp)

        println("Plotting")
        anim = @animate for (ti,t) in enumerate(ts)
            sol_ti = sol(t);
            xs, ys, temps = sol_ti[1:2:2N], sol_ti[2:2:2N], sol_ti[(4N+1):5N]

            xs .-= sum(xs)/N
            ys .-= sum(ys)/N

            lim = max(4, 1.5max(maximum(abs.(xs)),maximum(abs.(ys))))

            scatter(xs, ys, 
                title="$(Integer(round(10t))/10)", aspect_ratio=:equal,
                xlims=(-lim,lim), ylims=(-lim,lim), framestyle=:box,
                label="", 
                color=RGB(94/255, 79/255,162/255),
                dpi=150
                )
            scatter!(xs, ys, 
                label="", 
                markeralphas = [min(1, 2(temps[i]-mintemp) / (maxtemp-mintemp)) for i in eachindex(temps)],
                color=RGB(255/255,255/255,191/255)
                )
            scatter!(xs, ys, 
                label="", 
                markeralphas = [(max(midtemp,temps[i])-midtemp) / (maxtemp-midtemp) for i in eachindex(temps)],
                color=RGB(158/255,1/255,66/255)
                )

            # Insane bug fix with marker colours. GR is messed up, somehow.
            # It's not to do with the gif() saving via ffmpeg, it's GR's png
            # saving. Idk about other file formats.
            # It started with flickering in the gifs, replicable in series png
            # saving (see temp folder for gif), and has progressed to this insane
            # thing with RGB. Somehow, scaling from 0.25,1.0,0.25 to 0,1,0
            # produces the blue to red transition I want.
        end

        gif(anim, "planet_crash.gif", fps=15)


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
