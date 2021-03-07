    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Pre-processing: formatting LabChart data
% Processing:
% 1. EMG signals filtering
%   - pass-band filter 20Hz-450Hz
%   - FIR filter, method : Kaiser Window, beta = 6
% 2. EMG artefact detection and removing
%   - limiting signal to a threshold: mean + 3 * min(sd)
% threshold fixed to the mean of the period of interest + three times
% the minimum sd of sd values computed on 500ms' sliding windows over that period
% 3. Mean of absolute value of signal of interest (each portion)
%
% For the baseline condition:
%	threshold and mean calculated over the entire baseline condition (over 60s)
% For the overt speech, inner speech and listening conditions:
%	threshold and mean calculated over all the repetitions (default number is 6) of each word, so over 6s
%
% Input:
% ------
% for each subject,
% one .mat file for EMG data, one .mat file for audio data, from LabChart
% and one .csv file, from OpenSesame
% nb_repet: number of repetitions dof each word
% relax_time: duration in seconds of the relaxation session
% if show_emg = 1 : show emg channels (to check trigger synchronisation)
% if show_emg = 0 : don't show emg channels
% if show_audio = 1 : show audio file (to check trigger synchronisation)
% if show_audio = 0 : don't show audio file
%
% Output:
% -------
% 1. for each subject, two .txt files, contining mean of absolute EMG activity for each word in each condition

% 2. one figure with the filtered EMG signals for each channel and the wav file at the bottom
% relaxation boundaries and threshold displayed in green
% word boundaries and threshold in overt speech condition displayed in red
% word boundaries and threshold in inner speech condition displayed in blue
% word boundaries and threshold in listening condition displayed in black
%
% Author: H. Loevenbruck
% LPNC 15.05.2015
%
% Modifications: L. Nalborczyk
% LPNC 27.07.2016
% LPNC 29.03.2017
%
% Modifications: H. Loevenbruck
% LPNC 07.04.2017
%
% Use:
%---------------
% clear all
% close all
% cd /Users/Ladislas/Desktop/EMG
% cd /Users/hloeven/Helene/StagiairesEtCollab/LadislasNalborczyk/Manip_Zygoto_2017/OSF
% EMG_WAV_2017('S_06', 6, 60, 1000, 40000, 1, 1)
%
% To zoom in on a signal portion (horizontally):
% fig1 = get(figure(1));
% set(fig1.Children,'Xlim',[430 480]);
% To zoom out
% set(fig1.Children,'Xlim',[0 1400])
%
% To zoom in on vertical axis for audio only
% fig1 = figure(1);
% set(fig1.Children(size(fig1.Children,1)),'Ylim',[-0.5 0.5]);

function EMG_WAV_2017(num_sujet, nb_repet, relax_time, freq_emg, freq_wav, show_emg,  show_audio)

close all;

if (show_emg || show_audio)
    fig1 = figure(1);
    num_ax = [];     % saving EMG axes (plots) numbers
    axis_max = [];   % saving max size of plot for line drawings
end

pathname1 = pwd;
pathname = [pathname1 '/' num_sujet];

fname_emg = [num_sujet '_emg.mat'];
load([pathname '/' fname_emg]);
data_emg_tmp = data;
datastart_emg = datastart;
dataend_emg = dataend;
samplerate_emg = samplerate;
titles_emg = titles;
com_emg = com;

if show_audio
    fname_wav = [num_sujet '_wav.mat'];
    load([pathname '/' fname_wav]);
    data_wav_tmp = data;
	datastart_wav = datastart;
    dataend_wav = dataend;
    samplerate_wav = samplerate;
    titles_emg_wav = titles;
    com_wav = com;
end

fout1 = fopen(fullfile([num_sujet, '/', num_sujet, '_moyenne.txt']),'w');
fprintf(fout1,'%s\t%s\t%s\t%s\t%s\t%s\n', 'trigger_value', titles_emg(1,:), titles_emg(2,:), ...
    titles_emg(3,:), titles_emg(4,:), titles_emg(5,:));

frelax = fopen(fullfile([num_sujet, '/', num_sujet, '_relax.txt']),'w');
fprintf(frelax,'%s\t%s\t%s\t%s\t%s\n', titles_emg(1,:), titles_emg(2,:), ...
    titles_emg(3,:), titles_emg(4,:), titles_emg(5,:));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Formatting data obtained with LabChart %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%
% EMG data formatting
%%%%%%%%%%%%%%%%%%%%%
% Do EMG data before WAV
% to obtain the number of channels

[numchannels_emg numblocks_emg] = size(datastart_emg);

data_EMG = [];

for ch = 1:numchannels_emg,

    for bl = 1:numblocks_emg

        pdata = [];
        ptime = [];

        % Check sampling frequency
        if samplerate_emg(ch,bl) ~= freq_emg
            fprintf('Frequence d''echantillonage invalide\n');
            return
        end

        if (datastart_emg(ch,bl) ~= -1) % empty blocks excluded
            pdata = data_emg_tmp(datastart_emg(ch,bl):dataend_emg(ch,bl));

            if exist('scaleunits','var') % 16-bit data
                pdata = (pdata + scaleoffset(ch,bl)).* scaleunits(ch,bl);
            end

            ptime = [0:size(pdata,2)-1]/samplerate_emg(ch,bl);

            % Create a cell with all EMG data
            % Format: One column per channel and per block
            % Ch1_block1 Ch1_block2 Ch2_block1 Ch2_block2...
            % Blocks don't all have the same length
            indice_colonne = (ch-1)*numblocks_emg+ bl;
            data_EMG{indice_colonne} = pdata';

        end %if empty blocks excluded
    end % for all numblocks_emg
end % for all channels

mat_data_emg = cell2mat(data_EMG);

%%%%%%%%%%%%%%%%%%%%%
% WAV data formatting
%%%%%%%%%%%%%%%%%%%%%

if show_audio

    [numchannels_wav numblocks_wav] = size(datastart_wav);

    data_WAV = [];

    for ch = 1:numchannels_wav,

        for bl = 1:numblocks_wav

            pdata = [];
            ptime = [];

            % Check sampling frequency
            if samplerate_wav(ch,bl) ~= freq_wav
                fprintf('Frequence d''echantillonage invalide\n');
                return
            end

            if (datastart_wav(ch,bl) ~= -1) % empty blocks excluded
                pdata = data_wav_tmp(datastart_wav(ch,bl):dataend_wav(ch,bl));

                if exist('scaleunits','var') % 16-bit data
                    pdata = (pdata + scaleoffset(ch,bl)).* scaleunits(ch,bl);
                end

                % downsampling by 4 (from 40000 Hz to 10000Hz)
                wavdata = downsample(pdata,4);
                freq_wav = freq_wav/4;
                wavtime = [0:size(wavdata,2)-1]/freq_wav;

                % Plot audio data (below the last EMG channel)
                %subplot(numchannels_emg+1,1,numchannels_emg+1)
                subplot(3,1,3)
                plot(wavtime,wavdata), hold on
                title('AUDIO');
                xlabel('Time (s)');
                axes_audio = axis;

                % Create a cell with all EMG data
                % Format: One column per channel and per block
                % Ch1_block1 Ch1_block2 Ch2_block1 Ch2_block2...
                % Blocks don't all have the same length
                indice_colonne = (ch-1)*numblocks_wav+ bl;
                data_WAV{indice_colonne} = pdata';

            end %if empty blocks excluded
        end % for all numblocks_wav
    end % for all channels

    mat_data_wav = cell2mat(data_WAV);
end %endif show_audio

%%%%%%%%%%%%%%%%%%%%%
% Trigger formatting
%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% getting an ID for each muscle in each condition %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
com = com_emg;      % take all comments from the EMG mat file
com(:,3) = com(:,3) / 40; % convert comments from tickrate position to sample rate position

for i = 1:length(com) % replaces comments' indices by comments' values
    com(i,5) = str2num(comtext(com(i,5),:));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% find duplicated trigger values %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dup = [];

for i = 1:length(com)-1

    if ismember(0,com(i:i+1,5)) == 0;
        dup_temp = i;

        if exist('dup') == 0
                 dup = dup_temp;
        else
                dup = [dup ; dup_temp];
        end
    end
end

com(dup,:) = [];
com(find(com(:,5)==0),:)=[]; % removes 0 triggers from com

com_relax = com(1,:);
com(1,:)= []; % removes relax's com

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% imports trigger IDs from OpenSesame log file %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file = [num_sujet '/' num_sujet '.csv'];
fid = fopen(file,'r','n','UTF-8'); % reads log data from OpenSesame
logOS = textscan(fid,'%s','HeaderLines',1,'CollectOutput',1);
logOS = logOS{:};
fid = fclose(fid);
res = cellfun(@(x) strsplit(strrep(x, '"',''), ','), logOS, 'UniformOutput', false);

IDs = []; % initializes IDs' vector

for i = 1:length(res)
    IDs(i) = str2num(res{i}{3});
end

com(:,6) = IDs'; % create a 6th column for IDs  in 'com'

%%%%%%%%%%%%%%%%%%%%%%%
% EMG data processing %
%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%
% 1. Filtering %
%%%%%%%%%%%%%%%%

for ch = 1:numchannels_emg

    sig_emg = mat_data_emg(:,ch); % EMG signal of current muscle before filtering

    freq_dec = 1000;  % frequency after decimation : check ???

    %%%%%%%%%%%%%%%
    % comb filter %
    %%%%%%%%%%%%%%%

    fo = 50;
    q = 40;
    bw = (fo/(freq_dec/2))/q;

    [fcomb.b, fcomb.a] = iircomb(freq_dec/fo,bw,'notch');
    sig_emg = filter(fcomb.b,fcomb.a,sig_emg); % 50Hz notch filter

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Designing a pass-band filter %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % equivalent to the labchart one, i.e.,
    % Abs(Bandpass(chXX;10;300))*(1-Threshold(Abs(Bandpass(ChXX;10;300));Z))
    % where Z = mean(chXX) + 6 * sd(chXX)
    % FIR filter with Kaiser Window method with beta = 6
    % add notch filter (in first)

    fs = freq_emg;              % sampling rate
    fclpf = 450;                % high cut-off frequency
    initialDelaylpf = 0.064;    % initial online delay (in seconds)
    fchpf = 20;                 % low cut-off frequency
    initialDelayhpf = 0.5;      % initial online delay (in seconds)

    fclpfOfnyq = fclpf/(0.5*fs);
    fchpfOfnyq = fchpf/(0.5*fs);

    lenlpf = fix(initialDelaylpf*fs+1); % filter length in samples
    orderlpf = lenlpf-1;                % order
    blpf = fir1(orderlpf, fclpfOfnyq,'low',kaiser(lenlpf,6));

    lenhpf = fix(initialDelayhpf*fs+1); % filter length in samples
    orderhpf = lenhpf-1;                % order
    bhpf = fir1(orderhpf, fchpfOfnyq,'high',kaiser(lenhpf,6)); % coefs

    hd_blpf = dfilt.dffir(blpf);
    hd_bhpf = dfilt.dffir(bhpf);

    bp_emg = filter(hd_bhpf,sig_emg);
    bp_emg = filter(hd_blpf,bp_emg);
    %bp_emg = sig_emg;
    bp_emg_abs = abs(bp_emg);
    mat_data(:,ch) = bp_emg_abs; % mat_data contains absolute values of filtered EMG data for channel ch

    if show_emg
        if ch < 3 % only for OOI and ZYG
            % Plot filtered EMG data
            % compute time values for xaxis
            time_emg = [0:length(bp_emg)-1]/freq_emg;
            %subplot(numchannels_emg+1,1,ch)
            subplot(3,1,ch)
            plot(time_emg,bp_emg), hold on
            title(titles_emg(ch,:));
            xlabel('Time (s)');
            axis_max(ch, 1:4)=axis;
            num_ax  = [num_ax gca];  % save latest plot
        end

    end

end % for each muscle

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. managing and assembling triggers %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% relaxation period starts at the com_relax trigger and lasts for 60s (1mn)
relax_start = com_relax(3)/freq_emg; % ??? v???rifier, ??? un ???chantillon pr???s
relax_stop = relax_start + relax_time;
relax_data = mat_data(floor(com_relax(3)):floor(com_relax(3))+relax_time*freq_emg-1,:); % relaxation data

% listen, overt and inner periods start at each of the other com triggers
% and they each last for 1s
item_time = 1; % duree en secondes de chaque item de parole

% extract trigger values (which provide condition)
% overt speech condition : trigger values = 1 to 20
% inner speech condition : trigger values = 21 to 40
% listening condition : trigger values = 41 to 60
trigger_values = unique(com(:,6), 'stable'); % extract the 60 triggers IDs (in column 6 of "com")
trigger = zeros(nb_repet * item_time * freq_emg,length(titles_emg),length(trigger_values)); % initializes trigger matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. managing artefacts for each word portion
% A word portion consists of all the (6) reeptitions of each word.
% Depending on the trigger value, it can be in the
% overt speech, inner speech or listening condition.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Save time of all triggers associated with the same word
times_word = [];

% To create same y-axis limits for all graphs
min_ordinate = 0;
max_ordinate = 0;

% Concatenate nb_repet repetitions of each word
for i = 1:length(trigger_values) % make a loop for each trigger value (60 in total)

    % find all rows corresponding to the trigger
    % (same word in same condition)
    rows_trigger = find(com(:,6)==trigger_values(i));
    trigger_temp = zeros(1000,length(titles_emg));

    % for each word, in each condition
    for j = 1:length(rows_trigger) % 6 repetitions of each word in total

        % EMG data for each word in each condition
        trigger_temp(:,:) = mat_data(floor(com(rows_trigger(j),3)):floor(com(rows_trigger(j),3))+item_time*freq_emg-1,:);
        % Start time of the trigger associated with the word
        times_word(i,j) = com(rows_trigger(j),3)/freq_emg;

        if exist('trigger_temp1') == 0
             trigger_temp1 = trigger_temp;
        else
            trigger_temp1 = [trigger_temp1 ; trigger_temp]; % rowbinding six seconds of each word (6 repetitions)
        end
    end

    trigger(:,:,i) = trigger_temp1;
    clear trigger_temp1;

end %endfor all 60 triggers

mat_data_EMG = trigger; % 3D EMG data matrix (6s signals, 5 muscles, i_triggers) with only speech portions

for tr = 1:length(trigger_values) % 60 items (20 words in 3 conditions)

    tr_emg = mat_data_EMG(:,:,tr);      % concatenated EMG data for each word

    fprintf(fout1,'%d\t',trigger_values(tr));

    for ch = 1:length(titles_emg)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Threshold determination     %
        % On each concatenated period %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        bp_emg = tr_emg(:,ch);
        len_window = 0.5;
        nb_ech_sliding = len_window*freq_emg;
        h = nb_ech_sliding; % nb of samples in 500ms
        x = bp_emg;
        N = length(x);
        o = ones(1, N); % array for output

        for i = 1 : N
            % computes sd using the built-in std command
            % for the current window
            o(i) = std(x(max(1, i) : min(N, i + h - 1)));
        end

        o = o( 1 : h : length(o) ); % keeps only meaningful values
        o = o(o~=0); % removes zeros

        % threshold fixed to the 1-mn mean + three times the lowest sd (by windows of 500ms)
        seuil = mean(bp_emg)+3*min(o);
        % seuil = mean(bp_emg)+3*std(bp_emg);

        sig_emg = bp_emg;
        sig_emg(bp_emg>seuil) = NaN; % replaces value above the threshold by NaN
        emg_mean = mean(sig_emg(~isnan(sig_emg)));

        if show_emg
            % Plot trigger boundaries (where signal is taken from)
            % If overt speech: red boundaries and yellow threshold
            % If inner speech: blue boundaries and cyan threshold
            % If listen: black boundaries and grey threshold
            if ch < 3
            % Loop on all 6 repetitions for that trigger
            %subplot(numchannels_emg+1,1,ch)
            subplot(3,1,ch)
            axes_courant = axis_max(ch,1:4); % to plot vertical boundaries over entire plot
            for repet = 1: nb_repet
                    period_start = times_word(tr, repet);
                    period_end = period_start + item_time;
                    if trigger_values(tr) < 21  %overt speech
                        % Plot boundaries for each repetition
                        if mod(trigger_values(tr),2) == 0 % pair
                            text((period_start+ period_end)/2, axes_courant(4), 's');
                        else
                            text((period_start+ period_end)/2, axes_courant(4), 'r');
                        end
                            line([period_start period_start],[axes_courant(3) axes_courant(4)], 'color', 'r');
                        line([period_end period_end],[axes_courant(3) axes_courant(4)], 'color', 'r', 'LineStyle','--');
                        % Plot threshold, which was calculated over the 6
                        % repetitions
                        plot([period_start period_end], [seuil seuil], 'y');
                        plot([period_start period_end], [-seuil -seuil], 'y');
                    else if trigger_values(tr) < 41  %inner speech
                        % Plot boundaries for each repetition
                        if mod(trigger_values(tr),2) == 0 % pair
                            text((period_start+ period_end)/2, seuil, 's');
                        else
                            text((period_start+ period_end)/2, seuil, 'r');
                        end
                        line([period_start period_start],[axes_courant(3) axes_courant(4)], 'color', 'b');
                        line([period_end period_end],[axes_courant(3) axes_courant(4)], 'color', 'b', 'LineStyle','--');
                        % Plot threshold
                        plot([period_start period_end], [seuil seuil], 'c');
                        plot([period_start period_end], [-seuil -seuil], 'c');
                        else if trigger_values(tr) < 61  %listen
                        % Plot boundaries for each repetition
                        if mod(trigger_values(tr),2) == 0 % pair
                            text((period_start+ period_end)/2, seuil, 's');
                        else
                            text((period_start+ period_end)/2, seuil, 'r');
                        end
                        line([period_start period_start],[axes_courant(3) axes_courant(4)], 'color', 'k');
                        line([period_end period_end],[axes_courant(3) axes_courant(4)], 'color', 'k', 'LineStyle','--');
                        % Plot threshold
                        plot([period_start period_end], [seuil seuil], 'color', [0.7 0.7 0.7]);
                        plot([period_start period_end], [-seuil -seuil], 'color', [0.7 0.7 0.7]);
                            end
                        end
                    end
            end
            end
            %currentlim = ylim;
            %min_ordinate = min(min_ordinate, currentlim(1));
            %max_ordinate = max(max_ordinate, currentlim(2));
            min_ordinate = min(min_ordinate, -(mean(bp_emg)+4*min(o)));
            max_ordinate = max(max_ordinate, mean(bp_emg)+4*min(o));

        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % saving EMG mean for each channel
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(fout1,'%.9f\t',emg_mean);

    end %enfor channels
    fprintf(fout1,'\n');

    if show_emg
        % With linkaxes: The first axes (plot) you supply to linkaxes
        % determines the x- and y-limits for all linked axes
        linkaxes(num_ax,'y'); % link all EMG plots together so that they have same scale
        ylim ([min_ordinate max_ordinate]);
    end

    if show_audio
    % plot boundaries also on wav file
    % but only plot
        %subplot(numchannels_emg+1,1,numchannels_emg+1)
        subplot(3,1,3)
        for repet = 1: nb_repet
            period_start = times_word(tr, repet);
            period_end = period_start + item_time;
            if trigger_values(tr) < 21  %overt speech
                % Plot boundaries for each repetition
                line([period_start period_start],[axes_audio(3) axes_audio(4)], 'color', 'r');
                line([period_end period_end],[axes_audio(3) axes_audio(4)], 'color', 'r', 'LineStyle','--');
            else if trigger_values(tr) < 41  %inner speech
                % Plot boundaries for each repetition
                line([period_start period_start],[axes_audio(3) axes_audio(4)], 'color', 'b');
                line([period_end period_end],[axes_audio(3) axes_audio(4)], 'color', 'b', 'LineStyle','--');
                else if trigger_values(tr) < 61  %listen
                % Plot boundaries for each repetition
                line([period_start period_start],[axes_audio(3) axes_audio(4)], 'color', 'k');
                line([period_end period_end],[axes_audio(3) axes_audio(4)], 'color', 'k', 'LineStyle','--');
                    end
                end
            end
        end
    end

end %endfor all 60 triggers


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3. managing artefacts for the entire baseline session %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for ch = 1:length(titles_emg)

        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Threshold determination %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%

        bp_emg = relax_data(:,ch);

        len_window = 0.5;
        nb_ech_sliding = len_window*freq_emg;
        h = nb_ech_sliding; % nb of samples in 500ms

        x = bp_emg;
        N = length(x);

        o = ones(1, N); % array for output

        for i = 1 : N
            % computes sd using the built-in std command
            % for the current window
            o(i) = std(x(max(1, i) : min(N, i + h - 1)));
        end

        o = o( 1 : h : length(o) ); % keeps only meaningful values
        o = o(o~=0); % removes zeros

        % threshold fixed to the 1-mn mean + three times the lowest sd (by windows of 500ms)
        seuil = mean(bp_emg)+3*min(o);

        % seuil = mean(bp_emg)+3*std(bp_emg);

        sig_emg = bp_emg;
        sig_emg(bp_emg>seuil) = NaN; % replaces value above the threshold by threshold value

        emg_mean = mean(sig_emg(~isnan(sig_emg)));


        if show_emg
            if ch < 3
            % Plot relaxation boundaries in green
            %subplot(numchannels_emg+1,1,ch)
            subplot(3,1,ch)
            axes_courant = axis;
            line([relax_start relax_start],[axes_courant(3) axes_courant(4)], 'color', 'g');
            line([relax_stop relax_stop],[axes_courant(3) axes_courant(4)], 'color', 'g', 'LineStyle','--');

            % Plot threshold in green
            %subplot(numchannels_emg+1,1,ch)
            subplot(3,1,ch)
            plot([relax_start relax_stop], [seuil seuil], 'g');
            plot([relax_start relax_stop], [-seuil -seuil], 'g');
            %currentlim = ylim;
            %min_ordinate = min(min_ordinate, currentlim(1));
            %max_ordinate = max(max_ordinate, currentlim(2));
            end
        end

        if show_audio
            % Plot relaxation boundaries in green on wav signal
            %subplot(numchannels_emg+1,1,numchannels_emg+1)
            subplot(3,1,3)
            line([relax_start relax_start],[axes_audio(3) axes_audio(4)], 'color', 'g');
            line([relax_stop relax_stop],[axes_audio(3) axes_audio(4)], 'color', 'g', 'LineStyle','--');
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % saving EMG mean for relaxation
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fprintf(frelax,'%.9f\t',emg_mean);

end %endfor channels

fprintf(fout1,'\n');
fprintf(frelax,'\n');

fclose(frelax);
fclose(fout1);
