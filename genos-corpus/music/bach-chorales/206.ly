\version "2.10.33"

\header {
  title = "206 - So gibst du nun, mein Jesu, gute Nacht"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  \partial 4 bes4 
    d4. c8 bes4 d4 
  c4 bes4 bes4 a4 
  bes2 r4 d4 
  f4. ees8 d4 d8 c8 
  bes8 a8 g4 a4. g8   %5
  fis4 d4 r4 a'4 
  c4. bes8 a4 bes4 
  bes4 a4 g4. g8 
  a2 r4 a4 
  bes4 c4 d4 d8 e16 f16   %10
  e4. d8 cis4 d4 
  d4. cis8 d4 fis,4 
  g4 a4 bes4 g8 a16 bes16 
  a4. g8 fis4 g4 
  g4. fis8 g4   %15
  
}

alto = \relative c'' {
  \partial 4 g4 
    a8 g8 a4 g4 g4 
  g4 f4 ees8 d8 ees4 
  d2 r4 f4 
  f4. f8 f4 d4 
  d4 g8 f8 g4 ees4   %5
  d4 a4 r4 f'4 
  g8 f8 g4 f8 a8 g8 f8 
  e4 f2~ f8 e8 
  f2 r4 f4 
  f4 f4 f4 f4   %10
  bes8 a8 b4 a4 a4 
  a4 a4 a4 d,4 
  d4 d4 d4 g8 f8 
  ees8 d8 e4 d d
  d d d   %15
  
}

tenor = \relative c' {
  \partial 4 d4 
    d4 d4 d4 bes4 
  c4 d4 g,4 f8 ees8 
  f2 r4 bes4 
  c8 bes c4 bes a 
  g8 a8 bes4 ees8 d8 c8 bes8   %5
  a4 fis4 r4 d'4 
  c4 c4 c4 d4 
  g,4 a8 bes8 c4. c8 
  c2 r4 c4 
  d4 c4 bes4 d4   %10
  g8 f8 e4 e4 f4 
  e8 d8 e4 fis4 a,4 
  bes4 a4 g4 bes4 
  c8 bes8 a4 a4 bes4 
  a8 g8 a4 b  %15
  
}

baixo = \relative c' {
  \partial 4 g4 
    fis4. fis8 g4 g8 f8 
  ees4 d4 c4. c8 
  bes2 r4 bes'4 
  a4. a8 bes4 fis4 
  g8 f8 ees8 d8 c4 c4   %5
  d4 d,4 r4 d'4 
  e4. e8 f4 bes,4 
  c4 f4 c4. c8 
  f,2 r4 f'8 ees8 
  d8 c8 bes8 a8 bes4 bes8 a8   %10
  g4 gis4 a4 d4 
  a2 d4 d8 c8 
  bes8 a8 g8 fis8 g4 ees'8 d8 
  c4 cis4 d8 c8 bes8 g8 
  d'2 g,4   %15
  
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