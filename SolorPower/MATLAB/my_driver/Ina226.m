classdef Ina226 < realtime.internal.SourceSampleTime ...
        & coder.ExternalDependency ...
        & matlab.system.mixin.Propagates ...
        & matlab.system.mixin.CustomIcon
    %
    % System object for INA226 block.
    % 
    % Preperation
    %
    % - On Host PC
    %   $ pip install pigpio
    %
    % - On Remote Raspberry Pi
    %   $ sudo pigpiod
    %
    % Copyright 2019 Shogo MURAMATSU
    %
    %#codegen
    %#ok<*EMCA>
    
    properties
        % Public, tunable properties.
    end
    
    properties (Nontunable)
        % Public, non-tunable properties.
        %AddrStr = '192.168.11.21'
        %PortStr = '8888'
    end
    
    properties (Access = private)
        % Pre-computed constants.
        hIna226
    end
    
    methods
        % Constructor
        function obj = Ina226(varargin)
            % Support name-value pair arguments when constructing the object.
            setProperties(obj,nargin,varargin{:});
            %if isempty(coder.target)
            %     py.pigpio.pi(pyargs(obj.AddrStr,obj.PortStr))
            %else
            obj.hIna226 = coder.opaque('uint32_T','0');
            %end
        end
    end
    
    methods (Access=protected)
        function setupImpl(obj) 
            if isempty(coder.target)
                % Place simulation setup code here
                % remote connection?
            else
                % Call C-function implementing device initialization
                coder.cinclude('ina226_raspi.h');
                obj.hIna226 = coder.ceval('ina226Setup');
            end
        end
        
        function [v,c] = stepImpl(obj)   
            vword = int16(0);
            cword = int16(0);
            if isempty(coder.target)
                % Place simulation output code here
                % remote connection?
                v = double(0);
                c = double(0);
            else
                % Call C-function implementing device output
                vword = coder.ceval('readVoltage',obj.hIna226);
                cword = coder.ceval('readCurrent',obj.hIna226);                
                v = 1.25e-3*double(swapbytes(vword));
                c = 1e-3*double(swapbytes(cword));
            end
        end
        
        function releaseImpl(obj) 
            if isempty(coder.target)
                % Place simulation termination code here
            else
                % Call C-function implementing device termination
                coder.ceval('ina226Release',obj.hIna226);
            end
        end
    end
    
    methods (Access=protected)
        %% Define output properties
        function num = getNumInputsImpl(~)
            num = 0;
        end
        
        function num = getNumOutputsImpl(~)
            num = 2;
        end
        
        function flag = isOutputSizeLockedImpl(~,~)
            flag = true;
        end
        
        function varargout = isOutputFixedSizeImpl(~,~)
            varargout{1} = true;
            varargout{2} = true;
        end
        
        function flag = isOutputComplexityLockedImpl(~,~)
            flag = true;
        end
        
        function varargout = isOutputComplexImpl(~)
            varargout{1} = false;
            varargout{2} = false;            
        end
        
        function varargout = getOutputSizeImpl(~)
            varargout{1} = [1,1];
            varargout{2} = [1,1];            
        end
        
        function varargout = getOutputDataTypeImpl(~)
            varargout{1} = 'double';
            varargout{2} = 'double';            
        end
        
        function icon = getIconImpl(~)
            % Define a string as the icon for the System block in Simulink.
            icon = 'INA226';
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
            name = 'INA226';
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
                addSourceFiles(buildInfo,'ina226_raspi.c',srcDir);
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
