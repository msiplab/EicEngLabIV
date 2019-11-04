classdef Aqm0802 < matlab.System ...
        & coder.ExternalDependency ...
        & matlab.system.mixin.Propagates ...
        & matlab.system.mixin.CustomIcon
    %
    % System object for AQM0802 block.
    % 
    %
    
    % Copyright 2019 Shogo MURAMATSU
    %#codegen
    %#ok<*EMCA>
    
    properties
        % Public, tunable properties.
    end
    
    properties (Nontunable)
        % Public, non-tunable properties.
    end
    
    properties (Access = private)
        % Pre-computed constants.
        hAqm0802
    end
    
    methods
        % Constructor
        function obj = Aqm0802(varargin)
            % Support name-value pair arguments when constructing the object.
            setProperties(obj,nargin,varargin{:});
            obj.hAqm0802 = coder.opaque('uint32_T','0');
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj) 
            if isempty(coder.target)
                % Place simulation setup code here
            else
                % Call C-function implementing device initialization
                coder.cinclude('aqm0802_raspi.h');
                obj.hAqm0802 = coder.ceval('aqm0802Setup');
            end
        end
        
        function stepImpl(obj,line1,line2)  
            if isempty(coder.target)
                % Place simulation output code here
                %disp(line1)
                %disp(line2)
            else
                % Call C-function implementing device output
                len1 = length(line1);
                len2 = length(line2);
                coder.ceval('writeLine', obj.hAqm0802,0,line1,len1);
                coder.ceval('writeLine', obj.hAqm0802,1,line2,len2);
            end
        end
        
        function releaseImpl(obj) 
            if isempty(coder.target)
                % Place simulation termination code here
            else
                % Call C-function implementing device termination
                coder.ceval('aqm0802Release',obj.hAqm0802);
            end
        end
    end
    
    methods (Access=protected)
        %% Define input properties
        function num = getNumInputsImpl(~)
            num = 2;
        end
        
        function num = getNumOutputsImpl(~)
            num = 0;
        end
        
        function flag = isInputSizeLockedImpl(~,~)
            flag = true;
        end
        
        function varargout = isInputFixedSizeImpl(~,~)
            varargout{1} = true;
        end
        
        function flag = isInputComplexityLockedImpl(~,~)
            flag = true;
        end
        
        function validateInputsImpl(~, line1, line2)
            if isempty(coder.target)
                % Run input validation only in Simulation
                validateattributes(line1,{'uint8'},{'row'},'','line1');
                validateattributes(line2,{'uint8'},{'row'},'','line2');
            end
        end
        
        function icon = getIconImpl(~)
            % Define a string as the icon for the System block in Simulink.
            icon = 'AQM0802';
        end
    end
    
    methods (Static, Access=protected)
        function simMode = getSimulateUsingImpl(~)
            simMode = 'Interpreted execution';
        end
        
        function isVisible = showSimulateUsingImpl
            isVisible = false;
        end
    end
    
    methods (Static)
        function name = getDescriptiveName()
            name = 'AQM0802';
        end
        
        function b = isSupportedContext(context)
            b = context.isCodeGenTarget('rtw');
        end
        
        function updateBuildInfo(buildInfo, context)
            if context.isCodeGenTarget('rtw')
                % Update buildInfo
                srcDir = fullfile(fileparts(mfilename('fullpath')),'src'); 
                includeDir = fullfile(fileparts(mfilename('fullpath')),'include');
                addIncludePaths(buildInfo,includeDir);
                % Use the following API's to add include files, sources and
                % linker flags
                addSourceFiles(buildInfo,'aqm0802_raspi.c',srcDir);
                addCompileFlags(buildInfo,{'-pthread'});                
                addLinkFlags(buildInfo,{'-lpigpio', '-lrt'});                
                %addIncludeFiles(buildInfo,'source.h',includeDir);
                %addSourceFiles(buildInfo,'source.c',srcDir);
                %addLinkFlags(buildInfo,{'-lSource'});
                %addLinkObjects(buildInfo,'sourcelib.a',srcDir);
                %addCompileFlags(buildInfo,{'-D_DEBUG=1'});
                %addDefines(buildInfo,'MY_DEFINE_1')
            end
        end
    end
end
