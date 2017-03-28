\version "2.12.0"

pizz = ^\markup{\italic{pizz.}}
arco = ^\markup{\italic{arco}}

violoncelo= { 
	\clef bass 		
	\relative c, {
	 	
		
	c8 \p c' c  c c2:8 c2:8
		c2:8 c2:8 \crescTextCresc c2:8 \< 
		c2:8 c2:8 c2:8 
		c2:8 c2:8 bes2:8
		aes2:8 aes2:8 \< aes2:8 		
		ees2:8 ees2:8 ees2:8 \! 
	%7
		c4-. \f r g'4-. \f r  r2
		R1*3/2
		bes4-. \p r f'4-. r r2 
		R1*3/2
		aes,1. \p
		bes1.
		aes1.
		g1.
	%15
		g2 ~ (g4. b8 c2)
		aes4. (c8 f2 aes,4) r
		g2 ~ (g4. b8 c2)
		fis,4. (b8 c2 fis,4) r
		g4. (a8 \dimTextDim b2 \> ~ b4. cis8)
		d2 \> ~ d4. e8 \! fis2
		r2 fis r2		
	%22
		g, \< (ees d) \!
		c4. \< (d8 ees4. g8 c4. ees8) \!
 		ees2 \f (fis,4) r8 g (c4. ees8)
		ees2 (aes,4) r8 \crescTextCresc a  \< (c4. ees8)
		f2 (a,4) r8 bes (des4. f8)
	%27		
		aes4. c8-. \sf aes4. ees8-. c4. aes'8-. \sf 
		ees4. ces8-. aes4. ees'8-. \sf ces4. aes8-.
		ees4-. r ees'1->
		ges,1.->
		ees4-. \f r4 bes'4-. r4 r2
	%32
		ees,4-. r4 bes'4-. r4 r2
		R1*3/2
		R1*3/2 
		\clef tenor bes''1 \p (d,2) 
		bes'2 (c, d) \clef bass
	%37
		ees ges, \< bes 
		des, f aes, \!
		bes \f ces des
		ees f \> f, \!
		bes4 \p (ces'-. r aes-. r f-.)	
	%42
		r bes-. (r ges-. r ees-.)
		r8 des-. des-. g-. g-. fes-. fes-. bes-. bes-. g g des' 
		des4 (ces-.) (r ees-. r ees-.) \clef tenor
		r8 fis \p  fis gis a2 ~ (a8 gis) gis a
		r8 fis fis gis a2 ~ (a8 gis) gis a 
	%47		
		r8 \crescTextCresc a \< a a a gis gis a a2:8
		a2:8 \< a2:8 a2:8 \!
		R1*3/2 \clef bass
		r4 f,,4 (a d cis f)
		e (a,) r f' \< (e a,)
	%52	       
		r f' (e a,) \! r2
		aes \f aes'1
		g2 g, f
		f' g aes 
		bes bes, bes'
	%57
		aes aes,2.. aes'8
		g4. g,8 c,4. c'8 f,4. f'8
	 	bes,4. bes8 ees,4. ees'8 aes,4. aes8
		f4 r8 aes' f4. (c8) aes4. (f8) 
		ces'4. ( \dimTextDim f,8) \> ces'4. (f,8) ces'4. (f,8)	
	%62
		c'4. (f,8) \! c'4. (f,8) a4. \p (f8)
		bes4. (f8) c'4. (f,8) bes4. (ees,8)
		a4. (f8) c'4. (f,8) bes4. (ees,8)
		c'4. (f,8) bes4. (ees,8) c'4. \< (c,8)
		ees'4. (d,8) d'4. (g,8) g'4. \! (bes,8)			
	%67
		bes'2-. (c,-.) \p r2
		R1*3/2
		d2-. (ees-.) r2
		R1*3/2
		r2  c2-. \< (bes-.) \!
	%72
		a \< (bes) \! c-. \p
		(r2 bes-.) r2 
		bes1. 
		ees,4. \p  \< (f8 ges4. bes8 ees4. ges8) \!
		ges2 \> (a,4) \! r4 r2
	        bes2 \pizz r2 r2		 
	        bes2 r2 r2
	%79
		ees,4. \arco \< (f8 ges4. bes8 ees4. fis8) \!
	     	fis2 \> (fis,4) \! r4 r2
	        g2 \pizz r2 r2		 
	        g2 r2 r2
		fis'2 \p (fis,4) r4 r2
 		f'2 (f,4) r4 r2
		e'1 (c2 ~
		c2) aes1 \key c \major
	%84
		e1. \p 
		f
		d
		e
		e \< 
		\crescTextCresc f4. \< (g8 a4. c8 f4. a8)
	%90
		d,,1.
		e4. fis8 gis4. b8 e4 r
		a,4-. \f r r2 r2
		c,1 d2
            %colocar o marc.
		e4 r f'4. e8 d2
		c4 r f'4. e8 d2
	%96		
		r2  c,,4 \pizz \p r4 e r
	   	r2 f4 r4 a4 r4
		b4 r4 r2 r2
		b4 r4 r2 r2 \arco
		e4. \f fis8 g4 r4 r2
	    %colocar o marc.
	%101
		e4. fis8 g4. e8 c4 r4
		b4. g8 e2 fis
		g4 r4 c'4. b8 a2
		r2  g,4 \pizz \p r4 b r
		r2  c4 r4 e r
		fis4 r4 r2 ais4 r4
	%107 
		fis4 r4 r2 r2 
		d,4. \arco \p \< (fis8 b4. d8 fis4. g8 \!)
		g2 \> (b,4 \!) r4 r2
		ees,4. \< (g8 c4. ees8 g4. gis8 \!)
	%111
		gis2 \> (bis,4 \!) r4 r2  \key e \major
		e,4. \p ( \crescTextCresc fis8 \< gis4. cis8 e4. gis8)
		cis,,1.
		d4. \f (e8 fis4. a8 d4. fis8)
	%115
		gis,1.	
		d4. (e8 fis4. a8 d4. fis8)
		gis,1.	
		fis4. \f gis8 a2 r2
	%119
		b4. cis8 d2 r2
		dis4. e8 fis4. fisis8 gis4. a8
		fis4-. r4 gis r4 gis,-. r4
		r8 a'-. \f a-. gis-. r8 cis cis fis, fis b, b fis
		r8 d'' d cis r8 fis fis b, b e, e b \clef tenor
	%124
		r8 bis' bis cis r8 cis cis d r8 dis dis e
		r8 e e eis eis2 ~ eis8 e e f 
		f2 ~ f8 e e f f2 ~
		\crescTextCresc f8 \< (e) e4 ~ (e8 d) d4 ~ (d8 e,) \clef bass e4 ~
		(e8 d) d4 ~(d8 e,) e4 ~ (e8 d) d4
	%129
		cis4-. \f r r2 r8 a' cis, cis 
	   %adicionar o marc.
		fis4-. r4 r2 r8 a fis fis
		e4-. r4 r2 r8 a e e
		dis4-. r r2 r8 a' cis, cis  \key ees \major
	%133
		c4. \p (ees8 aes1)
		c,4. (ees8 aes1)
		c,1 (ees2 ~
		ees aes1)
		aes2:8 aes2:8 \< aes2:8
		\crescTextCresc a2:8 \< a2:8 g2:8
		fis2:8 fis2:8 f2:8
	%140
		ees2:8 ees2:8 des2:8
		c2 aes''-> ees-> 
		c-> aes-> ees->
		c4-. \f r4 g'-. r4 r2
		R1*3/2
		b4-. \p r4 fis'-. r4 r2 
		R1*3/2 
		<fis ais,>4-. \pizz r4 <fis b,>4-. r4 r2
		R1*3/2
	%149
		d,4 r e r fis r
		g r c r d r
		b1 \arco \p ~ (b4 f')
		ees1 ~ (ees4 c)
		b1 ~ (b4 a)
		g1. 
		g2 ~ (g4. b8 c2)
	%156
		aes4. (c8 f2 aes,)
		g2 ~ (g4. b8 c2) ~
		c2 (\dimTextDim c,1 \>) ~
		c1. \> ~ 
		c2. (d4 e2) \!
		r2 e-_ r2
	   %colocar o tenuto
	   	R1*3/2
		c'2 \p (\crescTextCresc bes2 \< g)
	%164
		f4. \< (g8 aes4. c8 f4. aes8) \!
		aes2 \f (b,4) r8 c8 (f4. aes8)
		aes2 (des,4) r8 d8 (\crescTextCresc f4. \< aes8)
		bes2 (d,4) r8 ees8 (ges4. bes8)
		des4. f8-. \sf des4. aes8 f4. des'8-. \sf 
	%169
		aes4. fes8 des4. a'8-. \sf fes4. des8
		aes4-. r4 aes'2:8  aes2:8
		c,2:8 c2:8 c2:8
		c,4-. r g'-. r r2 
		c,4-. r g'-. r r2 \clef tenor
		f''2 \p (d b)
	    %colocar o dolce
	%175
		c (ees fis)
		g1 (b,2)
		g'2 (a, b) \clef bass
		c ees, g
		bes, d f,
	%180
		g \f aes bes c d \> d, \!
		g4 \p (aes'-.) (r f-. r d-.)
		r g-. (r ees-. r c-.)
		r8 bes bes e e des des g g e e bes'
	%185
		bes4 (aes-.) (r c-. r c-.) \clef tenor
		r8 ees ees f ges2 ~ (ges8 f) f ges
		r8 ees ees f ges2 ~ (ges8 f) f ges
		r8 \crescTextCresc ges \< ges ges ges f f ges ges2:8
		ges2:8 \< ges2:8 ges2:8 \!
	%190
		R1*3/2 \clef bass
		r4 d,, \f (fis b ais d)
		cis (fis,) r4 d' (cis fis,)
		r4 d' \< (cis fis,) \! r2 
		f2 \f f'1
	%195
		e2 e, d 
		d' e f
		g g, g'
		f f,2. f'4
		e2 a,4. a'8 d,,4. d'8
	%200
		g,4. g'8 c,,4. c'8 f,4. f'8
		d,4 r8 f'8 d4. (a8) f4. (d8)
		aes'4. (\dimTextDim d,8 \>) aes'4. (d,8) \> aes'4. (d,8)   
		a'4. (d,8) a'4. (d,8) fis4. (d8) \p
	    %adicionar acima p dolce
	    	g4. (c,8) a'4. (d,8) g4. (c,8)
	%205   
		fis4. (d8) a'4. (d,8) g4. (c,8)
		a'4. (d,8) g4. (c,8) a'4. (a'8)
		<< c,4. ({ s4 \< s8 } >> b8) b4. (e8) e4. (g8)
		g,2-. \p (a2-.) r2 
		R1*3/2 
	%210
		b2-. (c2-.) r2 
		R1*3/2
		r2 a2  \< (g)
		fis (g) a-. \p 
		(r2 g2-.) r2
	%215
		g \pizz r2 r2 
		c,4. \arco (d8 \< ees4. g8 c4. ees8)
		ees2 \> (fis,4) \! r4 r2
		g2 \pizz r2 r2
		g2 r2 r2
		c,4. \arco  \p (d8 \< ees4. g8 c4. ees8)
	   %adicionar o piú p
		ees2 \> (fis,4) \! r4 r2
		g2 \pizz r2 r2
		g2 r2 r2 \time 2/2
	%224
		c,8 \p \crescTextCresc c' \< c c, c c' c c,  
	     %adicionar acima o crescendo ed agitato
	     	c c' c c,  c c' c c, 
		c c' c c,  c c' c c,
		c \< c' c c,  c c' c c, \!
		r4 bes'4-. \p (\crescTextCresc r4 \< e,4-.)
		r4 f-. (r aes-.) \clef tenor 
		r4 aes''-> ~ aes8 f f c
	%231
		b4-. \< aes2-> g4 \clef bass 
		c,,4. \f d8 e g c e
		e2 c
		f,4. g8 a c f a
		c2 c,
		f,4. \f g8 aes c f aes
		aes,4. c8 des f aes des
	%238
		g,,4-. g'' (aes g
		f d b g)
		f,4. g8 aes c f aes
		aes,4. c8 des f aes des
		g,,4-. g' (aes g
		f d b g)
		c,4. d8 e-. g-. c-. e-. 
		c,4. e8 f a c f
	%246
		c,4. e8 f aes des f 
		aes2 f
		c,4. d8 e g c e 
		c,4. e8 f a c f
		c,4. e8 f aes des f 
		aes2 f
	%252
		c,4. d8 e g c e 
		\times 2/3 {\dimTextDim c,2 \> d4} \times 2/3 {e4 g c}
	    	\times 2/3 {c,2 d4} \times 2/3 {e2 g4}
		\times 2/3 {c,2 d4} \times 2/3 {e2 g4}
		c,2 g' ~ 
		g c, ~
		c g' ~
		g c, ~
		<< c1 ^\fermata { s4 \p s4\< s4\! \> s4\! } >>
		
		
		 

  }
}
