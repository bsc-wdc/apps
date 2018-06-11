#!/usr/bin/perl

use strict;
use warnings;

my $bl2g = shift;
my $covin = shift;
my $idin = shift;
my $scaffoldfilt = shift;

if (!($bl2g && $covin && $idin && $scaffoldfilt)) {
	print STDERR "Error! Correct use: ./program bl2gFile coverage identity removeScaffolds?(Y/N)\n\n\n";
	die;
}


open BL2G, $bl2g;

my $cogiendo = 0;

my @entrada;

while (my $linea = <BL2G>) {
	chomp $linea;
	if ($linea =~ /coverage/) {
		# printamos el anterior si lo habia
		if ($cogiendo == 1) {
			foreach my $elem (@entrada) {
				print "$elem\n";
			}
		}

		@entrada = ();
		$linea =~ /coverage= +([^ ]+) /; 
		my $cov = $1;
		$linea =~ /identity= +([^ ]+) /;
		my $id = $1;
		#print "linea: $linea, cov: $cov, id: $id\n";
		if ( ($cov < $covin) || ($id < $idin) ) {
			$cogiendo = 0;
		}
		else {
			$cogiendo = 1;
		}
	}
	elsif ( ($linea =~ /scaffold/) && ($scaffoldfilt eq "Y") ) {
		$cogiendo = 0;
		@entrada = ();
	}

	if ($cogiendo == 1) {
		push @entrada, $linea;
	}

	if ($linea =~ /^##/) {
		if ($cogiendo == 1) {
			foreach my $elem (@entrada) {
				print "$elem\n";
			}
		}
		$cogiendo = 0;
		@entrada=();
	}

}




