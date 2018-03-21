#!/usr/bin/perl
#

 use Getopt::Long;
 
 &GetOptions("i=s" => \$reference,
	     "l=s" => \$ligand,
	     "r=s" => \$receptor,	
             "lig=s" =>\$lig,
             "rec=s" =>\$rec,
             "rdir=s" => \$rdir  
              );

$outlig=$ligand;
$outrec=$receptor;
$code[0]="ALA";
$code[1]="CYS";
$code[2]="ASP";
$code[3]="GLU";
$code[4]="PHE";
$code[5]="GLY";
$code[6]="HIS";
$code[7]="ILE";
$code[8]="LYS";
$code[9]="LEU";
$code[10]="MET";
$code[11]="ASN";
$code[12]="PRO";
$code[13]="GLN";
$code[14]="ARG";
$code[15]="SER";
$code[16]="THR";
$code[17]="VAL";
$code[18]="TRP";
$code[19]="TYR";

$lode="ACDEFGHIKLMNPQRSTVWY";


 
 open (REF,"<$reference")|| die ("Can't open input file:  $reference \n");
 $myout=substr($reference,0,4);
  $title=0;
 $NMR=0; 
 while (<REF>)
 {
 	@fields = split;
      if ($fields[0] eq "MODEL"){
         $chain=$fields[1];
         $NMR =1;
      }
      if ($fields[0] eq "ATOM"){
        $type=$fields[0];
	$type=~s/\s//g;
 	$atmnum=substr($_,6,5);
	$atmnum=~s/\s//g;
	$atomn=substr($_,12,4);
	$atomn=~s/\s//g;
	$res=substr($_,17,3);
	$res=~s/\s//g;
        if ($NMR == 0 ){
	$chain=substr($_,21,2);
	$chain=~s/\s//g;
        }
	$resnum=substr($_,23,8);
	$resnum=~s/\s//g;
	$coordx=substr($_,31,7);
	$coordx=~s/\s//g;
	$coordy=substr($_,39,7);
	$coordy=~s/\s//g;
	$coordz=substr($_,47,7);
	$coordz=~s/\s//g;
        $atom = {};
        $atom = {
           resnum => $resnum,
           residue=> $res,
           chain  => $chain,
           atom   => $atomn,
           atmnum => $atmnum,
           x      => $coordx,
           y      => $coordy,
           z      => $coordz,
          };
        push @prot,$atom;
        if ($res eq "HOH" || $res eq "WAT" || $res eq "H2O"  ) {break;}
      }else{next;};

 }
 close (REF);


   $outflush=select(PDB);
   $~ = "PDB_FORMAT";
   select($outflush);
   $outputPDBlig=$outlig;
    open (PDB,">$outputPDBlig");
   $n=1;
   $resnumw=0;
   $resnum_0=" ";
   for $i (0..$#prot)
   {
       $typew  ="ATOM";
       $chainw =$prot[$i]->{chain};
       $resnum = $prot[$i]->{resnum};
#       print "Chain $prot[$i]->{chain} Residue $resnum $resnum_0 $resnumw \n";
       if ($resnum ne $resnum_0){$resnumw++;$resnum_0=$resnum;} 
       $resw   = $prot[$i]->{residue};
       $atomw  = $prot[$i]->{atom};
       $atnumw = $n;
       @num=split /\./,$prot[$i]->{x};
       $x1 = $num[0];
       $x2 = substr($num[1],0,9);
       @num=split /\./,$prot[$i]->{y};
       $y1 = $num[0];
       $y2 = substr($num[1],0,9);
       @num=split /\./,$prot[$i]->{z};
       $z1 = $num[0];
       $z2 = substr($num[1],0,9);
       $n++;
       if(  $prot[$i]->{chain} =~ /[ $lig ]/ ) {write PDB;}
   }
   close PDB;

# receptor
   $outflush=select(PDB);
   $~ = "PDB_FORMAT";
   select($outflush);
   $outputPDBrec=$outrec;
    open (PDB,">$outputPDBrec");
   $n=1;
   $resnumw=0;
   $resnum_0=" ";
   for $i (0..$#prot)
   {
       $typew  ="ATOM";
       $chainw =$prot[$i]->{chain};
       $resnum = $prot[$i]->{resnum};
#       print "Chain $prot[$i]->{chain} Residue $resnum $resnum_0 $resnumw \n";
       if ($resnum ne $resnum_0){$resnumw++;$resnum_0=$resnum;} 
       $resw   = $prot[$i]->{residue};
       $atomw  = $prot[$i]->{atom};
       $atnumw = $n;
       @num=split /\./,$prot[$i]->{x};
       $x1 = $num[0];
       $x2 = substr($num[1],0,9);
       @num=split /\./,$prot[$i]->{y};
       $y1 = $num[0];
       $y2 = substr($num[1],0,9);
       @num=split /\./,$prot[$i]->{z};
       $z1 = $num[0];
       $z2 = substr($num[1],0,9);
       $n++;
       if(  $prot[$i]->{chain} =~ /[ $rec ]/ ) {write PDB;}
       if(  $prot[$i+1]->{chain} ne $prot[$i]->{chain} && $i+1 < $#prot ){$n=1;$resnum_0=" ";$resnumw=0;}
   }
   close PDB;
 # }


format PDB_FORMAT=
@<<<<<<@>>>  @<<<@<< @@>>>    @>>>.@<<@>>>.@<<@>>>.@<<
$typew,$atnumw,$atomw,$resw,$chainw,$resnumw,$x1,$x2,$y1,$y2,$z1,$z2
.
