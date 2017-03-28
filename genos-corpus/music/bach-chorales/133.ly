\version "2.10.33"

\header {
  title = "133 - Wir glauben all an einen Gott, Schöpfer Himmels und der Erden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c' {
  d2 a'4 g4 
  a4 e4 f2 
  e2 g2 
  f4 e4 d4 cis4 
  d2 a'4 a4   %5
  d4 c4 b4 a8 b8 
  c4 b4 a2 
  r4 b4 c4 a4 
  a4 e4 f4 f4 
  e2 d4 e4   %10
  f4 g4 a4 g4 
  f4 e4 d2 
  d'4 d4 e4 d4 
  e4 cis4 d2 
  d2 e4 b4   %15
  c4 a4 a4 e4 
  f2 e2 
  a4 gis4 a4 b4 
  c4 b4 a4 gis4 
  a2 b4 cis4   %20
  d2 a4 g4 
  a4 e4 f2 
  e2 g2 
  f4 e4 d4 cis4 
  d8 e8 f4 e2   %25
  d2 f4 g4 
  a4 b4 c4 b4 
  a2 r4 g4 
  f4 e4 d2 
  e2 f4 g4   %30
  f4 g4 d4 cis4 
  d1 
  
}

alto = \relative c {
  r4 a'4 d8 f4 e16 d16 
  e4. d16 cis16 d2~ 
  d4 cis4 r4 d8 e8~ 
  e8 d4 cis8 d16 c16 bes8 a4 
  a2 f'8 e8 d4~   %5
  d8 e16 f16 e4 e4. d8 
  e8 f4 e16 d16 c2 
  r4 e8 f8 g8 e8 c8 d8 
  e8 d8 e8 cis8 a8 b16 c16 d4~ 
  d8 cis16 b16 cis4 d4. cis8   %10
  a8 bes16 c16 d8 c8 c8 cis8 d8 e8~ 
  e8 d4 cis8 a2 
  b8 c8 d8 b8 c4 d8 g8 
  g8 a8 bes8 g8 a8 fis8 g4~ 
  g8 fis16 e16 fis4 e8 fis8 gis8 e8   %15
  a4 cis,4 d4. cis8 
  d4. cis16 b16 cis2 
  e2 e4 f4~ 
  f8 fis8 g8 gis8 a8 a,8 b8 d8~ 
  d8 c16 b16 c4 g'8 f8 e8 g8~   %20
  g8 f16 e16 f4 f4. e8~ 
  e8 d4 cis8 d2 
  e2 e4 a,4 
  d8 c8 bes8 c8 a8 bes8 g8 e'8 
  d8 cis8 d2 cis4   %25
  a2 d4 c4~ 
  c4 f8 d8 e8 f8 g8 e8 
  f2 r4 f8 e8~ 
  e8 d4 cis8 d2 
  cis8 d8 e4. d8 cis8 e8   %30
  a,8 bes4 a4 b8 a4 
  a1 
  
}

tenor = \relative c {
  r4 f8 g8 a4 d8 g,8~ 
  g8 f16 e16 a4 a2 
  a2 r4 d,8 a'8 
  a4 bes8 a16 g16 f8 g16 f16 e16 f16 g8~ 
  g8 f16 e16 f4 d'4 a4~   %5
  a8 gis8 a4 gis8 c4 b16 a16 
  gis8 a4 gis8 a2 
  r4 g8 f8 e8 g8 a4 
  a4. g8 f8. g16 a4 
  a2 f8 g16 a16 bes8 a16 g16   %10
  f4. e8 f8 e8 d8 bes'8 
  cis,8 a'8 bes8 a16 g16 f2 
  g2 g8 a8 b8 g8 
  c8 a8 e'4 d8 c4 bes8 
  a2 a4 e'8 d8   %15
  c8 d8 e4 a,2 
  a4. gis8 a2 
  a8 b8 c8 d8 c8 b8 a8 gis8 
  a4 e'4. d8 e4 
  e2 g,4 a4   %20
  a2 d4. c16 bes16 
  a2~ a8 g8 a8 b8 
  c2 cis8 d8 e4 
  a,4 g8 a8 fis8 g8 a4~ 
  a8 g8 a4 bes4 a8 g8~   %25
  g8 f16 e16 f4 bes8 a8 g8 c8 
  a4 d8 b8 g4. c8 
  c2 r4 d8 a8 
  a8 bes4 a16 g16 f2 
  a2 a8 bes4 a16 g16   %30
  a8 g16 f16 g8 f16 e16 f4 e16 f16 g8~ 
  g8 fis16 e16 fis2. 
  
}

baixo = \relative c {
  r4 d8 e8 f8 d8 bes'4 
  cis,4. b16 a16 d8 e8 f8 d8 
  a2 r4 b8 cis8 
  d8 f,8 g8 a8 bes8 g8 a4 
  d,2 d'8 e8 f8 d8   %5
  b4 c8 d8 e8 c8 f4 
  e8 d8 e4 a,2 
  r4 e'8 d8 c8 e8 f4 
  cis8 b8 cis8 a8 d8 a8 f8 d8 
  a'2 bes8 a8 g8 a8   %10
  d8 c8 bes8 c8 f,8 g16 a16 bes8 a16 g16 
  a8 f8 g8 a8 d,2 
  g8 a8 b8 g8 c4 g'8 b8 
  e,8 f8 g8 e8 fis8 d8 g4 
  d2 c8 d8 e4~   %15
  e8 f8 g8 e8 f8 g8 a8 g8 
  f8 e8 d4 a2 
  c8 d8 e8 b8 c4 d4~ 
  d8 dis8 e8 c8 f4 e4 
  a,2 e'4 a,4   %20
  d2 d8 c8 bes8 c8 
  f,8 g8 a4 d,8 e8 f8 d8 
  a'2 e'8 d8 cis8 e8 
  d4. c8 c8 bes8 a8 g8 
  f8 e8 d8 f8 g8 e8 a4   %25
  d,2 d'4 e4 
  f8 e8 d8 g8 e8 d8 e8 c8 
  f2 r4 b,8 cis8 
  d8 bes8 g8 a8 bes2 
  a8 b8 cis8 a8 d4 e4~   %30
  e8 d4 cis8 d8 gis,8 a4 
  d,1 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}