%clear all;
%close all;
function[] = scope_hysteresis_area(file_name,delta)
close all;
%a=load('scope_output.txt');
a=load(file_name);
%a(:,2)= a(:,2);
x_min = min(a(:,1));
y_min=min(a(:,2));
a(:,2)=a(:,2)-y_min;
a(:,1)=a(:,1)-x_min;
x_max = max(a(:,1));
y_max=max(a(:,2));
plot(a(:,1),a(:,2));
grid on;
hold on;
x_max = max(a(:,1));
x_min = min(a(:,1));
y_min=min(a(:,2));
%dx=0.1;
dx=delta;
N=ceil( (x_max-x_min)/dx)  ;
 %tmp_x_array(1)=10;
 index=1;
for i=1:N
    k=1;
    for j=1:size(a(:,1))
         if  ( (x_min+(i-1)*dx <=  a(j,1) )  & (  a(j,1) < x_min+i*dx ) )
                 
                  tmp_x_array(k)=a(j,2);             
                   k=k+1;
         end
              
    end

            
            upper_loop_max = max(tmp_x_array);
            lower_loop_min =min(tmp_x_array);
         if(size(upper_loop_max)~=1 | size(lower_loop_min)~=1  )
             error('ERROR: The steps are too small '); 
           
         end
            
            
            if(abs(upper_loop_max-lower_loop_min)>0.1 & (i < N-3)  & (  i > 3)   ) 
                new_x(index) = (x_min+i*dx + x_min+(i-1)*dx )/2;
                upper_loop(index) = upper_loop_max;
                lower_loop(index) = lower_loop_min;
                index=index+1;
                
            end
            
            if((i > N-3 )   | (  i < 3 )   )
                new_x(index) = (x_min+i*dx + x_min+(i-1)*dx )/2;
                upper_loop(index) = upper_loop_max;
                lower_loop(index) = lower_loop_min;
                index=index+1;
                
                
            end
          tmp_x_array(:)=[];
 end
plot(new_x,upper_loop,'r');
hold on;
plot(new_x,lower_loop,'g');


max_sec_derv = max(max(diff(lower_loop,2) ) ,max(diff(upper_loop,2)) ) ;
area_error = max_sec_derv*(x_max-x_min)^3 /(N*N);
area = trapz(new_x,upper_loop) - trapz(new_x,lower_loop);

str_area = num2str(area);
str_error = num2str(area_error);
Str = strcat('Area =', str_area,'\pm', str_error);

text((x_min),(y_max+y_min)/2,Str  ,'FontSize',15);


