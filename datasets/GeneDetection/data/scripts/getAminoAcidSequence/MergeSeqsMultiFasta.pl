



my $pathToflag = shift @ARGV;
my %t;
my $m;
my $numFasta = 0;
my $infile = shift @ARGV;
open(INFILE,"<".$infile);
$m=<INFILE>;
while( $m && $m =~ /\>(\S+)\s/)
    {
#    print "M -> $m\n";
    $numFasta++;
    $k = $1;
    my $seq = '';
    $m = <INFILE>;
    while( $m && $m !~ /\>(\S+)\s/)
      {
      chomp $m;
#      print "S -> $m\n";
      $seq = $seq . $m;
      $m = <INFILE>;    
      }
    $t{$k} = $seq;    
    } 
    
foreach $k (sort (keys %t))
    {
#    print $k . "\n";
    print $t{$k};# . "\n";
    }    
    
    
if($numFasta > 1) { 
    #print STDERR "$numFasta\n";
    qx[touch "$pathToflag/warn"];
    }    
 
 
       
