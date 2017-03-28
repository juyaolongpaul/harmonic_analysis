\version "2.10.33"

\header {
  title = "162 - Das alte Jahr vergangen ist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 g8 f8 e4 a4 
  g4 e4 f4 a4 
  a4 g8 f8 e4 a4 
  g4 e4 f4 a4 
  b4 cis4 d4 a8 b8   %5
  c4 b4 a4 b4 
  c4 b4 a4 a4 
  g8 f8 e4 f4 g4 
  g4 a4 g8 f8 e4 
  f4 f4 e4 b'4   %10
  c4 b4 a4 e4 
  a4 a4 gis4 
  
}

alto = \relative c' {
  \partial 4 e4 
    f4 e8 d8 cis4 d4 
  d8 e8 d8 cis8 d4 d4 
  d2~ d8 c8 c8 f8~ 
  f8 e16 d16 e8 cis8 d4 f4 
  f8 e8 e8 d8 d4 c8 d8   %5
  e4. d8 cis4 d4 
  e4 d4 c8 d8 ees4 
  d4 e4 d4 d8 e16 f16 
  e4 d8 c8 d4 e4~ 
  e8 cis8 d4 cis4 fis4   %10
  e8 fis8 gis8 a16 b16 e,4. e8 
  e4 dis4 e4 
  
}

tenor = \relative c' {
  \partial 4 cis4 
    d8 c8 bes8 g8 a4 a8 f8 
  d8 bes'8 a4 a4 f4 
  f8 d8 g4. e8 f8 a8 
  c8 bes8 bes8 a8 a4 d4 
  d4 a4 a8 gis8 a4   %5
  a4 gis4 e4 f4 
  g8 e8 f8 g8 a8 bes8 c4 
  bes8 a8 bes8 a8 a4 g4 
  c8 e,8 fis8 g16 a16 d,4 a'4 
  a4 a4 a4 a4   %10
  gis8 a8 b4 c8 d8 c8 b8 
  c8 a8 fis8 b8 b4 
  
}

baixo = \relative c' {
  \partial 4 a8  g8 
    f8 d8 g4. e8 f8 d8 
  bes8 g8 a4 d4 d8 c8 
  bes4 b4 c2~ 
  c8 e8 cis8 a8 d4 d'4 
  gis,4 g4 fis4 f4   %5
  e8 d8 e4 a,4 d4 
  c4 d8 e8 f4 fis4 
  g4 cis,4 d4 b4 
  c4. a8 b4 cis4 
  d8 e8 f8 g8 a4 dis,4   %10
  e4. d8 c8 b8 a8 g8 
  fis4 b4 e4 
  
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